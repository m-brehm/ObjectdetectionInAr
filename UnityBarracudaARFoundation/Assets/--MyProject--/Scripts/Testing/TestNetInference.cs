using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using Utils;
using System.Threading.Tasks;

public class TestNetInference : MonoBehaviour
{
    public NNModel modelFile;
    private IWorker worker;
    public string INPUT_NAME;
    public string OUTPUT_NAME;

    public int imageWidth;
    public int imageHeight;

    private bool isDetecting;
    // Start is called before the first frame update
    void Start()
    {
        //ServerManager.OnFrameReceived += OnCameraFrameReceived;
        var model = ModelLoader.Load(this.modelFile);
        //this.worker = GraphicsWorker.GetWorker(model);
        this.worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled ,model);
    }
    void OnDisable(){
        //ServerManager.OnFrameReceived -= OnCameraFrameReceived;
    }
    // Update is called once per frame
    void Update()
    {
        System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
        stopWatch.Start();
        TFDetect(stopWatch);
    }

    void OnCameraFrameReceived(){
        System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
        stopWatch.Start();
        TFDetect(stopWatch);
    }

    private void TFDetect(System.Diagnostics.Stopwatch stopWatch)
    {
        if (this.isDetecting)
        {
            return;
        }

        this.isDetecting = true;
        /*
        StartCoroutine(Detect(boxes=>
            {
                this.isDetecting = false;
                stopWatch.Stop();
                var ts = stopWatch.Elapsed;
                Debug.Log(ts);
            }));
            */
        //StartCoroutine(EvalCameraImageCoroutine(stopWatch));
        EvalCameraImage(stopWatch);
    }

    public IEnumerator Detect(System.Action<IList<BoundingBox>> callback){
        Color32[] picture = new Color32[imageWidth*imageHeight];
        var tensor = PreProcessing.TransformInput(picture, imageWidth, imageHeight);
        var inputs = new Dictionary<string, Tensor>();
        inputs.Add(INPUT_NAME, tensor);
        System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
        stopWatch.Start();
        //yield return StartCoroutine(worker.StartManualSchedule(inputs));
        worker.Execute(inputs);
        //Need to wait for worker here?
        Tensor output = worker.PeekOutput(OUTPUT_NAME);
        var boxes =  new List<BoundingBox>();
        tensor.Dispose();
        output.Dispose();
        callback(boxes);
        yield return null;
    }

    Tensor ExecuteInParts(IWorker worker, Tensor I, System.Diagnostics.Stopwatch stopWatch, int syncEveryNthLayer = 5)
    {
        var executor = worker.ExecuteAsync(I);
        var it = 0;
        bool hasMoreWork;

        do
        {
            hasMoreWork = executor.MoveNext();
            if (++it % syncEveryNthLayer == 0){
                worker.WaitForCompletion();
                //stopWatch.Stop();
                //var tsa = stopWatch.Elapsed;
                //Debug.Log(tsa);
            }

        } while (hasMoreWork);
        stopWatch.Stop();
        var ts = stopWatch.Elapsed;
        Debug.Log(ts);
        return worker.CopyOutput();
    }

    IEnumerator EvalCameraImageCoroutine(System.Diagnostics.Stopwatch stopWatch)
    {
        Color32[] picture = new Color32[imageWidth*imageHeight];
        var tensor = PreProcessing.TransformInput(picture, imageWidth, imageHeight);
        var it =  worker.StartManualSchedule(tensor);
        isDetecting = true;
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
        //Debug.Log(output.length);
        //var results = Utils.getBoundingBoxFromTensor(output, 1920, 1088);
        //results.ForEach(res => dh.log(res.ToString()));
        tensor.Dispose();
        output.Dispose();
        isDetecting = false;
        stopWatch.Stop();
        var ts = stopWatch.Elapsed;
        Debug.Log(ts);
    }

    void EvalCameraImage(System.Diagnostics.Stopwatch stopWatch)
    {
        Color32[] picture = new Color32[imageWidth*imageHeight];
        var tensor = PreProcessing.TransformInput(picture, imageWidth, imageHeight);
        var it =  worker.Execute(tensor);
        isDetecting = true;
        Tensor output = worker.PeekOutput(OUTPUT_NAME);
        //Debug.Log(output.length);
        //var results = Utils.getBoundingBoxFromTensor(output, 1920, 1088);
        //results.ForEach(res => dh.log(res.ToString()));
        tensor.Dispose();
        output.Dispose();
        isDetecting = false;
        stopWatch.Stop();
        var ts = stopWatch.Elapsed;
        Debug.Log(ts);
    }

    public void OnDestroy()
    {
        worker?.Dispose();
    }
}
