using System.Collections.Generic;
using System.Collections;
using Unity.Barracuda;
using UnityEngine;
using Utils;
using System.Threading.Tasks;
using System.Linq;



namespace ObjectDetection
{
    public class Yolov4tinyCPU : ObjectDetector
    {
        public override Tensor ProcessImage(Texture2D sourceTexture)
        {
            return PreProcessing.ProcessImage(sourceTexture, config.InputWidth);
        }
        public override void ProcessOutput()
        {
            Detections = new List<Detection>();
            var output_l = worker.PeekOutput("Identity");
            var output_m = worker.PeekOutput("Identity_1");
            //Debug.Log("Output: " + output_m);
            System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
            stopWatch.Start();
            Dictionary<string, float> scores = new Dictionary<string, float>();
            var results_l = ParseOutputs(output_l, 32, 13, scores);
            var results_m = ParseOutputs(output_m, 16, 26, scores);
            if(scores.Count>0){
                var keyOfMaxValue = scores.Aggregate((x, y) => x.Value > y.Value ? x : y).Key;
                Debug.Log(keyOfMaxValue+": "+scores[keyOfMaxValue]);
            }
            var results = results_l.Concat(results_m).ToList();


            //Debug.Log(ts);


            var boxes = PostProcessing.FilterBoundingBoxes(results, 5, threshold);
        }

        private IList<BoundingBox> ParseOutputs(Tensor yoloModelOutput, int cellSize, int featureMapSize,Dictionary<string, float> scores)
        {
            var boxes = new List<BoundingBox>();
            int count = 0;

            for (int cy = 0; cy < featureMapSize; cy++)
            {
                for (int cx = 0; cx < featureMapSize; cx++)
                {
                    for (int box = 0; box < 3; box++)
                    {
                        count++;
                        System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
                        stopWatch.Start();


                        var channel = (box * (80 + 5));
                        var bbd = PostProcessing.ExtractBoundingBoxDimensions(yoloModelOutput, cx, cy, channel);
                        float confidence = PostProcessing.GetConfidence(yoloModelOutput, cx, cy, channel);

                        if (confidence < threshold)
                        {
                            continue;
                        }

                        float[] predictedClasses = PostProcessing.ExtractClasses(yoloModelOutput, cx, cy, channel);
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

                        var mappedBoundingBox = PostProcessing.MapBoundingBoxToCell(cx, cy, box, bbd, cellSize, config.InputWidth, config.AnchorArray);
                        boxes.Add(new BoundingBox
                        {
                            Dimensions = new BoundingBoxDimensions
                            {
                                X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                                Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                                Width = mappedBoundingBox.Width,
                                Height = mappedBoundingBox.Height,
                            },
                            Confidence = topScore,
                            Label = labels[topResultIndex],
                            Used = false
                        });

                    }
                }
            }
            return boxes;
        }
    }
}
