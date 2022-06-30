using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System;
using System.Linq;
using System.Text.RegularExpressions;
using YoloV4Tiny;

public class WineBottleDetector : MonoBehaviour
{
    public NNModel model;
    public TextAsset labelsFile;
    IWorker _worker;
    private string[] labels;
    private bool isDetecting;
    private int InputWidth;
    private int ClassCount;
    private int FeatureMap1Width;
    private int FeatureMap2Width;
    private ComputeBuffer preprocess;
    public ComputeShader preprocessShader;
    [SerializeField] ImageSource _source = null;
    [SerializeField, Range(0, 1)] float _threshold = 0.5f;
    [SerializeField] Marker _markerPrefab = null;
    private List<Detection> detections;
    Marker[] _markers = new Marker[50];
    // Start is called before the first frame update
    void Start()
    {
        var _model = ModelLoader.Load(model);
        _worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, _model);
        //_worker = GraphicsWorker.GetWorker(_model);
        labels = Regex.Split(this.labelsFile.text, "\n|\r|\r\n")
            .Where(s => !String.IsNullOrEmpty(s)).ToArray();
        var inShape = _model.inputs[0].shape;
        var out1Shape = _model.GetShapeByName(_model.outputs[0]).Value;
        var out2Shape = _model.GetShapeByName(_model.outputs[1]).Value;
        InputWidth = inShape[6]; // 6: width
        //ClassCount = out1Shape.channels / AnchorCount - 5;
        FeatureMap1Width = out1Shape.width;
        FeatureMap2Width = out2Shape.width;
        preprocess = new ComputeBuffer(InputWidth * InputWidth * 3, sizeof(float));
        for (var i = 0; i < _markers.Length; i++)
            _markers[i] = Instantiate(_markerPrefab, _source.Preview.transform);
        
    }

    // Update is called once per frame
    void Update()
    {
        if(!isDetecting){
            System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
            stopWatch.Start();
            isDetecting=true;
            StartCoroutine(ProcessImage(_source.Texture, result =>{stopWatch.Stop();Debug.Log("Preprocess: "+stopWatch.Elapsed.TotalMilliseconds+"ms");stopWatch.Restart();
                //StartCoroutine(Detect(result,_threshold, detections =>{stopWatch.Stop();Debug.Log("Infernece: "+stopWatch.Elapsed.TotalMilliseconds+"ms");
                    Detect(result,_threshold);
                    stopWatch.Stop();Debug.Log("Infernece: "+stopWatch.Elapsed.TotalMilliseconds+"ms");
                    var i = 0;
                    foreach (var d in detections)
                    {
                        if (i == _markers.Length) break;
                        _markers[i++].SetAttributes(d);
                    }
                    for (; i < _markers.Length; i++) _markers[i].Hide();
                    isDetecting=false;
                //}));
            }));
        }
    }
    public IEnumerator Detect(Tensor t, float threshold, System.Action<IList<Detection>> result)
    {
        //var t = new Tensor(tex);
        // NN worker invocation
        detections = new List<Detection>();
		//yield return CoRoutineRunner.Instance.Run(_worker.StartManualSchedule(t));
        //var it = _worker.StartManualSchedule(t);
        _worker.Execute(t);
        /*
        int count = 0;
        while (it.MoveNext())
        {
          ++count;
          if (count % 20 == 0)
          {
            _worker.FlushSchedule(false);
            yield return null;
          }
        }
        _worker.FlushSchedule(true);
        */
        t.Dispose();
        //_worker.Execute(t);
        //_worker.FlushSchedule(true);
        var outputLayer0 = _worker.PeekOutput("boxes");
        var outputLayer1 = _worker.PeekOutput("confs");
        for(int i=0;i<400;i++){
            if(outputLayer1[0,0,0,i]>threshold){
                var detection = new Detection();
                detection.x=outputLayer0[0,0,0,i];
                detection.y=outputLayer0[0,0,1,i];
                detection.w=outputLayer0[0,0,2,i];
                detection.h=outputLayer0[0,0,3,i];
                detection.score = outputLayer1[0,0,0,i];
                detection.classIndex = 20;
                detections.Add(detection);
                /*
                Debug.Log(outputLayer1[0,0,0,i]+" "+i);
                Debug.Log(outputLayer0[0,0,0,i]);
                Debug.Log(outputLayer0[0,0,1,i]);
                Debug.Log(outputLayer0[0,0,2,i]);
                Debug.Log(outputLayer0[0,0,3,i]);
                */
            }
        }

        result(adjustDetections(detections));
        yield return null;
    }
    public void Detect(Tensor t, float threshold)
    {
        detections = new List<Detection>();
        _worker.Execute(t);
        t.Dispose();
        var outputLayer0 = _worker.PeekOutput("boxes");
        var outputLayer1 = _worker.PeekOutput("confs");
        for(int i=0;i<400;i++){
            if(outputLayer1[0,0,0,i]>threshold){
                var detection = new Detection();
                detection.x=outputLayer0[0,0,0,i];
                detection.y=outputLayer0[0,0,1,i];
                detection.w=outputLayer0[0,0,2,i];
                detection.h=outputLayer0[0,0,3,i];
                detection.score = outputLayer1[0,0,0,i];
                detection.classIndex = 20;
                detections.Add(detection);
            }
        }
        detections = adjustDetections(detections);
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

    
    public IEnumerator ProcessImage(Texture sourceTexture, System.Action<Tensor> result){
        var pre = preprocessShader;
        //var tex = new Texture2D(_config.InputWidth,_config.InputWidth);
        var tex = RTUtil.NewFloat(InputWidth,InputWidth);
        tex.enableRandomWrite = true;
        pre.SetInt("Size", InputWidth);
        pre.SetTexture(0, "Image", sourceTexture);
        pre.SetTexture(0,"Result",tex);
        pre.SetBuffer(0, "Tensor", preprocess);
        pre.DispatchThreads(0, InputWidth, InputWidth, 1);
        var t = new Tensor(new TensorShape(1, InputWidth, InputWidth, 3), preprocess);
        result(t);
        //result(Utils.PreProcessing.toTexture2D(tex));
        yield return null;
    }
}
