using System;
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Collections;
using ObjectDetection;
using Unity.Barracuda;
using System.Threading;

sealed class Visualizer : MonoBehaviour
{
    #region Editable attributes
    public enum Detectors{
        Yolo2_tiny,
        Yolo4TinyPP,
        Yolo4TinyCPU,
        Yolo4TinyAR,
        Yolo4TinyGPU
    };
    [SerializeField] Detectors selected_detector;
    [SerializeField] ImageSource _source = null;
    [SerializeField] Marker _markerPrefab = null;

    [SerializeField]Text preProcessTimeUI;
    [SerializeField]Text preProcessTimeAvg;
    [SerializeField]Text inferenceTimeUI;
    [SerializeField]Text inferenceTimeAvg;
    [SerializeField]Text postProcessTimeUI;
    [SerializeField]Text postProcessTimeAvg;
    [SerializeField]Text progress;
    double preProcessTime = 0;
    double inferenceTime = 0;
    double postProcessTime = 0;
    double preProcessTimeCounter = 0;
    double inferenceTimeCounter = 0;
    double postProcessTimeCounter = 0;
    int currentFrame = 0;

    #endregion

    #region Internal objects

    ObjectDetector _detector;
    
    Marker[] _markers = new Marker[50];

    float timer = 0f;

    #endregion

    #region MonoBehaviour implementation

    void Start()
    {
        if (selected_detector == Detectors.Yolo2_tiny)
        {
            _detector = null;//GameObject.Find("Detector Yolo2-tiny").GetComponent<DetectorYolo2>();
        }
        else if (selected_detector == Detectors.Yolo4TinyPP)
        {
            _detector = GameObject.Find("Yolov4tinyPP").GetComponent<Yolov4tinyPP>();
        }
        else if (selected_detector == Detectors.Yolo4TinyCPU)
        {
            _detector = GameObject.Find("Yolov4tinyCPU").GetComponent<Yolov4tinyCPU>();
        }
        else if (selected_detector == Detectors.Yolo4TinyAR)
        {
            _detector = GameObject.Find("Yolov4tinyAR").GetComponent<Yolov4tinyAR>();
        }
        else if (selected_detector == Detectors.Yolo4TinyGPU)
        {
            _detector = GameObject.Find("Yolov4tinyGPU").GetComponent<Yolov4TinyGPU>();
        }
        else
        {
            Debug.Log("DEBUG: Invalid detector model");
        }
        //_detector = new ObjectDetector(_resources);
        _detector.Init();
        for (var i = 0; i < _markers.Length; i++)
            _markers[i] = Instantiate(_markerPrefab, _source.Preview.transform);
    }

    void OnDisable()
      => _detector.Destroy();

    void OnDestroy()
    {
        for (var i = 0; i < _markers.Length; i++) Destroy(_markers[i]);
    }

/*
    void RunInferenceAsync(){
        if(!_detector.Detecting)
            StartCoroutine(_detector.ProcessImageCoRoutine(_source.Texture, result =>{
                StartCoroutine(_detector.DetectCoRoutine(result,_threshold, detections =>{
                    var i = 0;
                    Debug.Log("There are "+detections.Count+" Detections");
                    foreach (var d in detections)
                    {
                        if (i == _markers.Length) break;
                        _markers[i++].SetAttributes(d);
                    }
                    for (; i < _markers.Length; i++) _markers[i].Hide();
                    //_source.nextImage();
                }));
            }));
    }

    void RunInference(){
        if(!_detector.Detecting)
            StartCoroutine(_detector.ProcessImageCoRoutine(_source.Texture, result =>{
                    IList<Detection> detections = _detector.Detect(result);
                    var i = 0;
                    Debug.Log("There are "+detections.Count+" Detections");
                    foreach (var d in detections)
                    {
                        if (i == _markers.Length) break;
                        _markers[i++].SetAttributes(d);
                    }
                    for (; i < _markers.Length; i++) _markers[i].Hide();
                    //_source.nextImage();
            }));
    }
    */

    void benchmark(){
        if(!_detector.Detecting){
            _source.nextImage();
            currentFrame++;
            _detector.benchmark(_source.Texture2D,out preProcessTime,out inferenceTime,out postProcessTime);
            preProcessTimeUI.text = preProcessTime+"ms";
            preProcessTimeCounter+=preProcessTime;
            preProcessTimeAvg.text = preProcessTimeCounter/currentFrame+"ms";
            inferenceTimeUI.text = inferenceTime+"ms";
            inferenceTimeCounter+=inferenceTime;
            inferenceTimeAvg.text = inferenceTimeCounter/currentFrame+"ms";
            postProcessTimeUI.text = postProcessTime+"ms";
            postProcessTimeCounter+=postProcessTime;
            postProcessTimeAvg.text = postProcessTimeCounter/currentFrame+"ms";
            var i = 0;
            //Debug.Log("There are "+detections.Count+" Detections");
            foreach (var d in _detector.Detections)
            {
                if (i == _markers.Length) break;
                _markers[i++].SetAttributes(d);
            }
            foreach (var label in _source._dataset.images[_source.currentImageIndex].labels)
            {
                Detection d = new Detection();
                d.x = label.rect.center.x/_source._texture.width;
                d.y = 1-label.rect.center.y/_source._texture.height;
                d.w = (label.rect.xMax-label.rect.xMin)/_source._texture.width;
                d.h = (label.rect.yMax-label.rect.yMin)/_source._texture.height;
                d.score = 1;
                d.classIndex = (uint)label.classIndex;
                if (i == _markers.Length) break;
                _markers[i++].SetAttributes(d);
            }
            progress.text = currentFrame+"";
            for (; i < _markers.Length; i++) _markers[i].Hide();
            Debug.Log("There are "+_detector.Detections.Count+" Detections");
        }

    }

    void Update()
    {
        //RunInference();
        benchmark();
    }

    #endregion
}
