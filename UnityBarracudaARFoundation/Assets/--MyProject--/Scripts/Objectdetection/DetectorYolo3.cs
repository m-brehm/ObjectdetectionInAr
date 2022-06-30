using System;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using Utils;
public class DetectorYolo3 : MonoBehaviour, Detector
{

    public NNModel modelFile;
    public TextAsset labelsFile;


    // ONNX model input and output name. Modify when switching models.
    //These aren't const values because they need to be easily edited on the component before play mode

    public string INPUT_NAME;
    public string OUTPUT_NAME_L;
    public string OUTPUT_NAME_M;

    //This has to stay a const
    private const int _image_size = 416;
    public int IMAGE_SIZE { get => _image_size; }

    // Minimum detection confidence to track a detection
    public float MINIMUM_CONFIDENCE = 0.10f;

    private IWorker worker;

    // public const int ROW_COUNT_L = 13;
    // public const int COL_COUNT_L = 13;
    // public const int ROW_COUNT_M = 26;
    // public const int COL_COUNT_M = 26;
    public Dictionary<string, int> params_l = new Dictionary<string, int>(){{"ROW_COUNT", 13}, {"COL_COUNT", 13}, {"CELL_WIDTH", 32}, {"CELL_HEIGHT", 32}};
    public Dictionary<string, int> params_m = new Dictionary<string, int>(){{"ROW_COUNT", 26}, {"COL_COUNT", 26}, {"CELL_WIDTH", 16}, {"CELL_HEIGHT", 16}};
    public const int BOXES_PER_CELL = 3;
    public const int BOX_INFO_FEATURE_COUNT = 5;

    //Update this!
    public int CLASS_COUNT;

    // public const float CELL_WIDTH_L = 32;
    // public const float CELL_HEIGHT_L = 32;
    // public const float CELL_WIDTH_M = 16;
    // public const float CELL_HEIGHT_M = 16;
    private string[] labels;

    private float[] anchors = new float[]
    {
        10F, 14F,  23F, 27F,  37F, 58F,  81F, 82F,  135F, 169F,  344F, 319F // yolov3-tiny
    };


    public void Start()
    {
        this.labels = Regex.Split(this.labelsFile.text, "\n|\r|\r\n")
            .Where(s => !String.IsNullOrEmpty(s)).ToArray();
        var model = ModelLoader.Load(this.modelFile);
        // https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Worker.html
        //These checks all check for GPU before CPU as GPU is preferred if the platform + rendering pipeline support it
        this.worker = GraphicsWorker.GetWorker(model);
    }




    public IEnumerator Detect(Color32[] picture, System.Action<IList<BoundingBox>> callback)
    {
        using (var tensor = PreProcessing.TransformInput(picture, IMAGE_SIZE, IMAGE_SIZE))
        {
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add(INPUT_NAME, tensor);
            yield return StartCoroutine(worker.StartManualSchedule(inputs));
            //worker.Execute(inputs);
            var output_l = worker.PeekOutput(OUTPUT_NAME_L);
            var output_m = worker.PeekOutput(OUTPUT_NAME_M);
            //Debug.Log("Output: " + output_m);
            System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
            stopWatch.Start();
            Dictionary<string, float> scores = new Dictionary<string, float>();
            var results_l = ParseOutputs(output_l, MINIMUM_CONFIDENCE, params_l,scores);
            var results_m = ParseOutputs(output_m, MINIMUM_CONFIDENCE, params_m,scores);
            if(scores.Count>0){
                var keyOfMaxValue = scores.Aggregate((x, y) => x.Value > y.Value ? x : y).Key;
                Debug.Log(keyOfMaxValue+": "+scores[keyOfMaxValue]);
            }
            var results = results_l.Concat(results_m).ToList();


            stopWatch.Stop();
            TimeSpan ts = stopWatch.Elapsed;
            //Debug.Log(ts);


            var boxes = PostProcessing.FilterBoundingBoxes(results, 5, MINIMUM_CONFIDENCE);
            callback(boxes);
            yield return null;
        }
    }

    private IList<BoundingBox> ParseOutputs(Tensor yoloModelOutput, float threshold, Dictionary<string, int> parameters,Dictionary<string, float> scores)
    {
        var boxes = new List<BoundingBox>();
        double averageTime = 0;
        int count = 0;

        for (int cy = 0; cy < parameters["COL_COUNT"]; cy++)
        {
            for (int cx = 0; cx < parameters["ROW_COUNT"]; cx++)
            {
                for (int box = 0; box < BOXES_PER_CELL; box++)
                {
                    count++;
                    System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
                    stopWatch.Start();


                    var channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));
                    var bbd = ExtractBoundingBoxDimensions(yoloModelOutput, cx, cy, channel);
                    float confidence = GetConfidence(yoloModelOutput, cx, cy, channel);

                    if (confidence < threshold)
                    {
                        continue;
                    }

                    float[] predictedClasses = ExtractClasses(yoloModelOutput, cx, cy, channel);
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

                    var mappedBoundingBox = MapBoundingBoxToCell(cx, cy, box, bbd, parameters);
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

                    stopWatch.Stop();
                    TimeSpan ts = stopWatch.Elapsed;
                    averageTime+=ts.TotalMilliseconds;
                    //Debug.Log("time per box "+ts);
                }
            }
        }
        //Debug.Log("averageTime: "+averageTime/count);
        return boxes;
    }

    private BoundingBoxDimensions ExtractBoundingBoxDimensions(Tensor modelOutput, int x, int y, int channel)
    {
        return new BoundingBoxDimensions
        {
            X = modelOutput[0, x, y, channel],
            Y = modelOutput[0, x, y, channel + 1],
            Width = modelOutput[0, x, y, channel + 2],
            Height = modelOutput[0, x, y, channel + 3]
        };
    }


    private float GetConfidence(Tensor modelOutput, int x, int y, int channel)
    {
        //Debug.Log("ModelOutput " + modelOutput);
        return PostProcessing.Sigmoid(modelOutput[0, x, y, channel + 4]);
    }


    private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions, Dictionary<string, int> parameters)
    {
        return new CellDimensions
        {
            X = ((float)y + PostProcessing.Sigmoid(boxDimensions.X)) * parameters["CELL_WIDTH"],
            Y = ((float)x + PostProcessing.Sigmoid(boxDimensions.Y)) * parameters["CELL_HEIGHT"],
            Width = (float)Math.Exp(boxDimensions.Width) * anchors[6 + box * 2],
            Height = (float)Math.Exp(boxDimensions.Height) * anchors[6 + box * 2 + 1],
        };
    }


    public float[] ExtractClasses(Tensor modelOutput, int x, int y, int channel)
    {
        float[] predictedClasses = new float[CLASS_COUNT];
        int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;

        for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
        {
            predictedClasses[predictedClass] = modelOutput[0, x, y, predictedClass + predictedClassOffset];
        }

        return PostProcessing.Softmax(predictedClasses);
    }
}
