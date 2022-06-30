using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CoRoutineRunner : MonoBehaviour
{
    public static CoRoutineRunner Instance { get; private set; }
 
    void Awake()
    {
        Instance = this;
    }
 
    public Coroutine Run(IEnumerator cor)
    {
        return StartCoroutine(cor);
    }
}
