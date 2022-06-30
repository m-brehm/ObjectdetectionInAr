using System.Collections.Generic;
using System.Collections;
using Unity.Barracuda;
using UnityEngine;
using Utils;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using System.Linq;


namespace ObjectDetection
{
    public class Yolov4tinyAR : ObjectDetector
    {
        public ComputeShader preprocessCompute;
        ComputeBuffer preprocess;
        public override void Init()
        {
            base.Init();
            preprocess = new ComputeBuffer(config.InputFootprint, sizeof(float));
        }
        public override Tensor ProcessImage(Texture2D sourceTexture)
        {
            //var tex = new Texture2D(config.InputWidth,config.InputWidth);
            var tex = RTUtil.NewFloat(config.InputWidth, config.InputWidth);
            tex.enableRandomWrite = true;
            preprocessCompute.SetInt("Size", config.InputWidth);
            preprocessCompute.SetTexture(0, "Image", sourceTexture);
            preprocessCompute.SetTexture(0, "Result", tex);
            preprocessCompute.SetBuffer(0, "Tensor", preprocess);
            preprocessCompute.DispatchThreads(0, config.InputWidth, config.InputWidth, 1);
            var t = new Tensor(config.InputShape, preprocess);
            return t;
            //result(Utils.PreProcessing.toTexture2D(tex));
        }
        public override void ProcessOutput()
        {
            Detections = new List<Detection>();
            var output_l = worker.PeekOutput("Identity");
            var output_m = worker.PeekOutput("Identity_1");
            //Debug.Log("Output: " + output_m);
            Dictionary<string, float> scores = new Dictionary<string, float>();
            var results_l = ParseOutputs(output_l, config.FeatureMap1Width, scores);
            var results_m = ParseOutputs(output_m, config.FeatureMap2Width, scores);
            if(scores.Count>0){
                var keyOfMaxValue = scores.Aggregate((x, y) => x.Value > y.Value ? x : y).Key;
                Debug.Log(keyOfMaxValue+": "+scores[keyOfMaxValue]);
            }
            var results = results_l.Concat(results_m).ToList();
            Debug.Log(results.Count);

            Detections = adjustDetections(PostProcessing.FilterBoundingBoxes(results, 5, threshold));
        }

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

        private IList<Detection> ParseOutputs(Tensor yoloModelOutput, int featureMapSize,Dictionary<string, float> scores)
        {
            var boxes = new List<Detection>();
            int count = 0;

            for (int cy = 0; cy < featureMapSize; cy++)
            {
                for (int cx = 0; cx < featureMapSize; cx++)
                {
                    for (int box = 0; box < 3; box++)
                    {
                        count++;
                        var channel = (box * (80 + 5));
                        var bbd = PostProcessing.ExtractBoundingBoxDimensions(yoloModelOutput, cx, cy, channel);
                        float confidence = PostProcessing.GetConfidence(yoloModelOutput, cx, cy, channel);

                        if (confidence < threshold)
                        {
                            continue;
                        }

                        float[] predictedClasses = PostProcessing.ExtractClasses(yoloModelOutput, cx, cy, channel, 80 ,5);
                        var (topResultIndex, topResultScore) = PostProcessing.GetTopResult(predictedClasses);
                        var topScore = topResultScore * confidence;
                        //Debug.Log("DEBUG: results: " + labels[topResultIndex]);
                        string label = labels[topResultIndex];
                        if(!scores.ContainsKey(label))
                            scores.Add(label,confidence);
                        else if(scores[label]<confidence)
                            scores[label]=confidence;

                        if (topScore < threshold)
                        {
                            continue;
                        }

                        var mappedBoundingBox = PostProcessing.MapBoundingBoxToCell(cx, cy, box, bbd, featureMapSize, config.InputWidth, config.AnchorArray);
                        boxes.Add(new Detection
                        {
                  
                            x = mappedBoundingBox.X,
                            y = mappedBoundingBox.Y,
                            w = mappedBoundingBox.Width,
                            h = mappedBoundingBox.Height,
                            score = topScore,
                            classIndex = (uint)topResultIndex
                        });

                    }
                }
            }
            return boxes;
        }

        public void ProcessOutputParallel()
        {
            Detections = new List<Detection>();
            float[] output1 = worker.PeekOutput("Identity").ToReadOnlyArray();
            NativeArray<float> output1_n = new NativeArray<float>(output1, Allocator.TempJob);
            float[] output2 = worker.PeekOutput("Identity_1").ToReadOnlyArray();
            NativeArray<float> output2_n = new NativeArray<float>(output2, Allocator.TempJob);
            NativeArray<float> anchors_n = new NativeArray<float>(config.AnchorArray, Allocator.TempJob);
            NativeList<Detection> detections1_n= new NativeList<Detection>(Allocator.TempJob);
            NativeList<Detection> detections2_n= new NativeList<Detection>(Allocator.TempJob);
            var processOutput1 = new ProcessOutputJob
            {
                anchors = anchors_n,
                detections = detections1_n.AsParallelWriter(),
                output = output1_n,
                threshold = threshold
            };
            var processOutput1Handle = processOutput1.Schedule(output1_n.Length/255,1);

            var processOutput2 = new ProcessOutputJob
            {
                anchors = anchors_n,
                detections = detections2_n.AsParallelWriter(),
                output = output2_n,
                threshold = threshold
            };
            var processOutput2Handle = processOutput2.Schedule(output2_n.Length/255,1);
            processOutput1Handle.Complete();
            processOutput2Handle.Complete();
            foreach(Detection detection in detections1_n)
                Detections.Add(detection);
            foreach(Detection detection in detections2_n)
                Detections.Add(detection);
            detections1_n.Dispose();
            detections2_n.Dispose();
            anchors_n.Dispose();
        }


        [BurstCompile(CompileSynchronously = true)]
        public struct ProcessOutputJob : IJobParallelFor
        {
            //[NativeDisableParallelForRestriction] public NativeArray<Vector3> normals;
            [ReadOnly] public NativeArray<float> anchors;
            [NativeDisableParallelForRestriction] public NativeList<Detection>.ParallelWriter detections;
            [ReadOnly][DeallocateOnJobCompletion] public NativeArray<float> output;
            //[ReadOnly][DeallocateOnJobCompletion] public NativeArray<BoneWeight> boneWeights;
            [ReadOnly][DeallocateOnJobCompletion] public float threshold;

            public void Execute(int index)
            {
                for(int i=0;i<3;i++){
                    if(output[index*255+i*85]<threshold){
                        continue;
                    }

                }
                
            }
        }

    }
}
