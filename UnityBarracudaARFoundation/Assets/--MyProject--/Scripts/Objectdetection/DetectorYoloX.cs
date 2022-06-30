using System;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using Utils;
using System.Threading.Tasks;
public class DetectorYoloX : MonoBehaviour, Detector
{

    public NNModel modelFile;
    public TextAsset labelsFile;

    private const int IMAGE_MEAN = 0;
    private const float IMAGE_STD = 255.0F;

    // ONNX model input and output name. Modify when switching models.
    //These aren't const values because they need to be easily edited on the component before play mode

    public string INPUT_NAME;
    public string OUTPUT_NAME;

    //This has to stay a const
    private const int _image_size = 416;
    public int IMAGE_SIZE { get => _image_size; }

    // Minimum detection confidence to track a detection
    public float MINIMUM_CONFIDENCE = 0.10f;
    public int dimensions = 85; // 0,1,2,3 ->box,4->confidence，5-85 -> coco classes confidence 
    //private int rows = size / dimensions; //25200
    public int confidenceIndex = 4;
    public int labelStartIndex = 5;

    private IWorker worker;

    // public const int ROW_COUNT_L = 13;
    // public const int COL_COUNT_L = 13;
    // public const int ROW_COUNT_M = 26;
    // public const int COL_COUNT_M = 26;
    public Dictionary<string, int> params_l = new Dictionary<string, int>(){{"ROW_COUNT", 13}, {"COL_COUNT", 3549}, {"CELL_WIDTH", 32}, {"CELL_HEIGHT", 32}};
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
        //this.worker = GraphicsWorker.GetWorker(model);
        this.worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled,model);
    }




    public IEnumerator Detect(Color32[] picture, System.Action<IList<BoundingBox>> callback)
    {
        using (var tensor = PreProcessing.TransformInput(picture, IMAGE_SIZE, IMAGE_SIZE))
        {
            var inputs = new Dictionary<string, Tensor>();
            inputs.Add(INPUT_NAME, tensor);
            var it =  worker.StartManualSchedule(tensor);
            int count = 0;

            while (it.MoveNext())
            {
                ++count;
                if (count % 20 == 0)
                {
                    Task.Run(() => worker.FlushSchedule(false));
                    yield return null;
                }
            }
            worker.FlushSchedule(true);
            Tensor output = worker.PeekOutput(OUTPUT_NAME);
            Dictionary<string, float> scores = new Dictionary<string, float>();
            var results = ParseOutputs(output, MINIMUM_CONFIDENCE, scores);
            //var results = Utils.getBoundingBoxFromTensor(output, 1920, 1088);
            //results.ForEach(res => dh.log(res.ToString()));
            if(scores.Count>0){
                var keyOfMaxValue = scores.Aggregate((x, y) => x.Value > y.Value ? x : y).Key;
                Debug.Log(keyOfMaxValue+": "+scores[keyOfMaxValue]);
            }
            tensor.Dispose();
            output.Dispose();
            //var boxes =  new List<BoundingBox>();
            var boxes = PostProcessing.FilterBoundingBoxes(results, 5, MINIMUM_CONFIDENCE);
            callback(boxes);
        }
    }

    private IList<BoundingBox> ParseOutputs(Tensor output, float threshold,Dictionary<string, float> scores)
    {
        var boxes = new List<BoundingBox>();

        for (int b = 0; b < output.length/dimensions; b++)
        {
            float confidence = output[0, 0, 4, b];
            if (confidence < threshold)
            {
                //continue;
            }
            var bbd = PostProcessing.ExtractBoundingBoxDimensions(output, b);
            float[] predictedClasses = PostProcessing.ExtractClasses(output, b, CLASS_COUNT, BOX_INFO_FEATURE_COUNT);
            var (topResultIndex, topResultScore) = PostProcessing.GetTopResult(predictedClasses);
            var topScore = topResultScore * confidence;
            //Debug.Log("DEBUG: results: " + labels[topResultIndex]);
            string label = labels[topResultIndex];
            if(!scores.ContainsKey(label))
                scores.Add(label,topScore);
            else if(scores[label]<topScore)
                scores[label]=topScore;
            if (topScore < threshold)
            {
                continue;
            }
            //var mappedBoundingBox = PostProcessing.MapBoundingBoxToCell(b, bbd);
            boxes.Add(new BoundingBox
            {
                Dimensions = new BoundingBoxDimensions
                {/*
                    X = (bbd.X - bbd.Width / 2),
                    Y = (bbd.Y - bbd.Height / 2),
                    Width = bbd.Width,
                    Height = bbd.Height,
                    */
                    X = bbd.X,
                    Y = bbd.Y,
                    Width = bbd.Width-bbd.X,
                    Height = bbd.Height-bbd.Y ,
                },
                Confidence = topScore,
                Label = labels[topResultIndex],
                Used = false
            });
        }
        return boxes;
    }
}
