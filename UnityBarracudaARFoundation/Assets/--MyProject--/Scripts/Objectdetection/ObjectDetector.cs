using System.Collections.Generic;
using System.Collections;
using Unity.Barracuda;
using UnityEngine;
using Utils;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.Linq;
using System;

namespace ObjectDetection
{

    public abstract class ObjectDetector : MonoBehaviour
    {
        #region setup
        [SerializeField] NNModel model_file;
        [SerializeField] TextAsset labels_file;
        [SerializeField] TextAsset anchors_file;
        [SerializeField, Range(0, 1)] protected float threshold = 0.5f;
        [SerializeField] WorkerFactory.Type worker_type;
        #endregion

        #region private properties
        protected static string[] labels;
        protected Config config;
        protected IWorker worker;
        #endregion

        #region Public properties
        [HideInInspector]public bool Detecting { get => detecting; }
        [HideInInspector]public bool detecting = false;
        public IList<Detection> Detections;
        #endregion

        public virtual void Init()
        {
            labels = labels_file.text.Split('\n');
            var anchors_strings = anchors_file.text.Split('\n');
            var anchors = new float[anchors_strings.Length];
            for(int i=0; i<anchors_strings.Length;i++){
                float.TryParse(anchors_strings[i], out anchors[i]);
            }
            var model = ModelLoader.Load(model_file);
            config = new Config(anchors, model);
            worker = WorkerFactory.CreateWorker(worker_type, model);
        }

        public virtual void Destroy(){
            worker?.Dispose();
            worker = null;
        }

        

        #region Main inference function

        public IEnumerator DetectCoRoutine(Tensor t, float threshold, System.Action<IList<Detection>> result)
        {
            //var t = new Tensor(tex);
            // NN worker invocation
            detecting = true;
            yield return CoRoutineRunner.Instance.Run(worker.StartManualSchedule(t));
            //_worker.Execute(t);
            /*
            var it = _worker.StartManualSchedule(t);
            int count = 0;
            detecting = true;
            while (it.MoveNext())
            {
                ++count;
                if (count % 20 == 0)
                {
                    Task.Run(() => _worker.FlushSchedule(false));
                    yield return null;
                }
            }
            _worker.FlushSchedule(true);
            t.Dispose();
            */
            // NN output retrieval
            ProcessOutput();
            result(Detections);

        }

        public IList<Detection> Detect(Tensor t)
        {
            worker.Execute(t);
            t.Dispose();
            ProcessOutput();
            return Detections;
        }

        #endregion

        #region Single Steps
        public abstract Tensor ProcessImage(Texture2D sourceTexture);
        public abstract void ProcessOutput();
        public void benchmark(Texture2D sourceTexture, out double preProcessTime, out double inferenceTime, out double postProcessTime)
        {
            detecting = true;
            System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();


            stopWatch.Start();
            Tensor input = ProcessImage(sourceTexture);
            stopWatch.Stop();
            preProcessTime = stopWatch.ElapsedMilliseconds;


            stopWatch.Restart();
            worker.Execute(input);
            stopWatch.Stop();
            inferenceTime = stopWatch.ElapsedMilliseconds;


            stopWatch.Restart();
            ProcessOutput();
            stopWatch.Stop();
            postProcessTime = stopWatch.ElapsedMilliseconds;
            detecting = false;

        }
        #endregion
    }

}
