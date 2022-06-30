using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;
public class Test : MonoBehaviour
{
    public NNModel model_file;
    public WorkerFactory.Type worker_type;
    public Text time;
    private IWorker worker;
    private Model model;
    // Start is called before the first frame update
    void Start()
    {
        model = ModelLoader.Load(model_file);

        worker = WorkerFactory.CreateWorker(worker_type, model);
    }

    // Update is called once per frame
    void Update()
    {
        System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch();
        Tensor input = new Tensor(model.inputs[0].shape);
        stopWatch.Start();
        worker.Execute(input);
        stopWatch.Stop();
        time.text = stopWatch.ElapsedMilliseconds+"";
        input.Dispose();
    }
}
