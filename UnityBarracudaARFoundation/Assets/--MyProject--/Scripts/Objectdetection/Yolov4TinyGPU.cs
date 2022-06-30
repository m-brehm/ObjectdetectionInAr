using System.Collections.Generic;
using System.Collections;
using Unity.Barracuda;
using UnityEngine;
using Utils;
using System.Threading.Tasks;

namespace ObjectDetection
{
    public class Yolov4TinyGPU : ObjectDetector
    {
		(ComputeBuffer preprocess,
         RenderTexture feature1,
         RenderTexture feature2,
         ComputeBuffer post1,
         ComputeBuffer post2,
         ComputeBuffer counter,
         ComputeBuffer countRead) _buffers;
        DetectionCache _readCache;

		[SerializeField] ComputeShader preprocessCompute, postprocess1Compute, postprocess2Compute;
        public override void Init()
        {
			base.Init();
            // NN model loading
            // Buffer allocation
            _buffers.preprocess = new ComputeBuffer
              (config.InputFootprint, sizeof(float));

            _buffers.feature1 = RTUtil.NewFloat
              (config.FeatureDataSize, config.FeatureMap1Footprint);

            _buffers.feature2 = RTUtil.NewFloat
              (config.FeatureDataSize, config.FeatureMap2Footprint);

            _buffers.post1 = new ComputeBuffer
              (Config.MaxDetection, Detection.Size);

            _buffers.post2 = new ComputeBuffer
              (Config.MaxDetection, Detection.Size, ComputeBufferType.Append);

            _buffers.counter = new ComputeBuffer
              (1, sizeof(uint), ComputeBufferType.Counter);

            _buffers.countRead = new ComputeBuffer
              (1, sizeof(uint), ComputeBufferType.Raw);

            // Detection data read cache initialization
            _readCache = new DetectionCache(_buffers.post2, _buffers.countRead);
        }

        public override void Destroy()
        {
			base.Destroy();
            _buffers.preprocess?.Dispose();
            _buffers.preprocess = null;

            ObjectUtil.Destroy(_buffers.feature1);
            _buffers.feature1 = null;

            ObjectUtil.Destroy(_buffers.feature2);
            _buffers.feature2 = null;

            _buffers.post1?.Dispose();
            _buffers.post1 = null;

            _buffers.post2?.Dispose();
            _buffers.post2 = null;

            _buffers.counter?.Dispose();
            _buffers.counter = null;

            _buffers.countRead?.Dispose();
            _buffers.countRead = null;
        }


        public IEnumerator ProcessImageCoRoutine(Texture sourceTexture, System.Action<Tensor> result)
        {
            //var tex = new Texture2D(config.InputWidth,config.InputWidth);
            var tex = RTUtil.NewFloat(config.InputWidth, config.InputWidth);
            tex.enableRandomWrite = true;
            preprocessCompute.SetInt("Size", config.InputWidth);
            preprocessCompute.SetTexture(0, "Image", sourceTexture);
            preprocessCompute.SetTexture(0, "Result", tex);
            preprocessCompute.SetBuffer(0, "Tensor", _buffers.preprocess);
            preprocessCompute.DispatchThreads(0, config.InputWidth, config.InputWidth, 1);
            var t = new Tensor(config.InputShape, _buffers.preprocess);
            result(t);
            //result(Utils.PreProcessing.toTexture2D(tex));
            yield return null;
        }
        
		public override Tensor ProcessImage(Texture2D sourceTexture)
        {
            //var tex = new Texture2D(config.InputWidth,config.InputWidth);
            var tex = RTUtil.NewFloat(config.InputWidth, config.InputWidth);
            tex.enableRandomWrite = true;
            preprocessCompute.SetInt("Size", config.InputWidth);
            preprocessCompute.SetTexture(0, "Image", sourceTexture);
            preprocessCompute.SetTexture(0, "Result", tex);
            preprocessCompute.SetBuffer(0, "Tensor", _buffers.preprocess);
            preprocessCompute.DispatchThreads(0, config.InputWidth, config.InputWidth, 1);
            var t = new Tensor(config.InputShape, _buffers.preprocess);
            return t;
            //result(Utils.PreProcessing.toTexture2D(tex));
        }
        
		public Tensor ProcessImageCPU(Texture2D sourceTexture)
        {
            return PreProcessing.ProcessImage(sourceTexture, config.InputWidth);
        }

		public override void ProcessOutput()
        {
            worker.CopyOutput("Identity", _buffers.feature1);
            worker.CopyOutput("Identity_1", _buffers.feature2);

            // Counter buffer reset
            _buffers.post2.SetCounterValue(0);
            _buffers.counter.SetCounterValue(0);

            // First stage postprocessing: detection data aggregation
            postprocess1Compute.SetInt("ClassCount", config.ClassCount);
            postprocess1Compute.SetFloat("Threshold", threshold);
            postprocess1Compute.SetBuffer(0, "Output", _buffers.post1);
            postprocess1Compute.SetBuffer(0, "OutputCount", _buffers.counter);

            // (feature map 1)
            var width1 = config.FeatureMap1Width;
            postprocess1Compute.SetTexture(0, "Input", _buffers.feature1);
            postprocess1Compute.SetInt("InputSize", width1);
            postprocess1Compute.SetFloats("Anchors", config.AnchorArray1);
            postprocess1Compute.DispatchThreads(0, width1, width1, 1);

            // (feature map 2)
            var width2 = config.FeatureMap2Width;
            postprocess1Compute.SetTexture(0, "Input", _buffers.feature2);
            postprocess1Compute.SetInt("InputSize", width2);
            postprocess1Compute.SetFloats("Anchors", config.AnchorArray2);
            postprocess1Compute.DispatchThreads(0, width2, width2, 1);

            // Second stage postprocessing: overlap removal
            postprocess2Compute.SetFloat("Threshold", 0.5f);
            postprocess2Compute.SetBuffer(0, "Input", _buffers.post1);
            postprocess2Compute.SetBuffer(0, "InputCount", _buffers.counter);
            postprocess2Compute.SetBuffer(0, "Output", _buffers.post2);
            postprocess2Compute.Dispatch(0, 1, 1, 1);

            // Bounding box count after removal
            ComputeBuffer.CopyCount(_buffers.post2, _buffers.countRead, 0);

            // Cache data invalidation
            _readCache.Invalidate();
            Detections = adjustDetections(_readCache.Cached);
            detecting = false;
            //yield return null;
        }

		//Adjust Detections to UI Maker
        private List<Detection> adjustDetections(IList<Detection> detections)
        {
            List<Detection> adjustedDetections = new List<Detection>();
            foreach (Detection d in detections)
            {
                Detection adjustedDetection = new Detection();
                adjustedDetection.x = d.x;
                adjustedDetection.y = (1 - d.y);
                adjustedDetection.w = d.w;
                adjustedDetection.h = d.h;
                adjustedDetection.classIndex = d.classIndex;
                adjustedDetection.score = d.score;
                adjustedDetections.Add(adjustedDetection);
            }

            return adjustedDetections;
        }
    }
}