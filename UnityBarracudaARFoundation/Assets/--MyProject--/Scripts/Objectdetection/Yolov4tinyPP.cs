using System.Collections.Generic;
using System.Collections;
using Unity.Barracuda;
using UnityEngine;
using Utils;
using System.Threading.Tasks;

namespace ObjectDetection
{
    public class Yolov4tinyPP : ObjectDetector
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
            var outputLayer0 = worker.PeekOutput("boxes");
            var outputLayer1 = worker.PeekOutput("confs");
            for(int i=0;i<2535;i++){
                if(outputLayer1[0,0,0,i]>threshold){
                    var detection = new Detection();
                    detection.x=outputLayer0[0,0,0,i];
                    detection.y=outputLayer0[0,0,1,i];
                    detection.w=outputLayer0[0,0,2,i];
                    detection.h=outputLayer0[0,0,3,i];
                    var scores = PostProcessing.ExtractClasses(outputLayer1,i,80);
                    var (topResultIndex, topResultScore) = PostProcessing.GetTopResult(scores);
                if(topResultScore>threshold){
                    detection.score = topResultScore;
                    detection.classIndex = (uint)topResultIndex;
                    Detections.Add(detection);
                }
                }
        }
        Detections = adjustDetections(PostProcessing.FilterBoundingBoxes(Detections, 30, threshold));
        }

        private List<Detection> adjustDetections(IList<Detection> detections){
            List<Detection> adjustedDetections = new List<Detection>();
            foreach(Detection d in detections){
                Detection adjustedDetection = new Detection();
                adjustedDetection.x = (d.x+(d.w-d.x)/2);
                adjustedDetection.y = (1-(d.y+(d.h-d.y)/2));
                adjustedDetection.w = (d.w-d.x);
                adjustedDetection.h = (d.h-d.y);
                adjustedDetection.classIndex = d.classIndex;
                adjustedDetection.score = d.score;
                adjustedDetections.Add(adjustedDetection);
            }

            return adjustedDetections;
        }
    }
}
