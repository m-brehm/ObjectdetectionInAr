using UnityEngine;
using UnityEngine.UI;
using YoloV4Tiny;
using System.Text.RegularExpressions;
using System.Linq;
using System;
sealed class Marker : MonoBehaviour
{
    public Rect _rect;
    RectTransform _parent;
    RectTransform _xform;
    Image _panel;
    Text _label;
    public TextAsset _labelsFile;

    public static string[] _labels = new[]
    {
        "Plane", "Bicycle", "Bird", "Boat",
        "Bottle", "Bus", "Car", "Cat",
        "Chair", "Cow", "Table", "Dog",
        "Horse", "Motorbike", "Person", "Plant",
        "Sheep", "Sofa", "Train", "TV", "Winebottle"
    };

    void Start()
    {
        _labels = Regex.Split(_labelsFile.text, "\n|\r|\r\n")
            .Where(s => !String.IsNullOrEmpty(s)).ToArray();
        _xform = GetComponent<RectTransform>();
        _parent = (RectTransform)_xform.parent;
        _panel = GetComponent<Image>();
        _label = GetComponentInChildren<Text>();
    }

    public void SetAttributes(in Detection d)
    {
        var rect = _parent.rect;

        // Bounding box position
        /*
        var x = d.x * rect.width;
        var y = (1 - d.y) * rect.height;
        var w = d.w * rect.width;
        var h = d.h * rect.height;
        */
        /*
        var x = d.x * rect.width;
        var y = (1 - d.y) * rect.height;
        var w = d.w * rect.width;
        var h = (1-d.h) * rect.height;

        var x = (d.x+(d.w-d.x)/2) * rect.width;
        var y = (1-(d.y+(d.h-d.y)/2)) * rect.height;
        var w = (d.w-d.x) * rect.width;
        var h = (d.h-d.y) * rect.height;
        */
        var x = d.x * rect.width;
        var y = d.y * rect.height;
        var w = d.w * rect.width;
        var h = d.h * rect.height;
        

        
        _xform.anchoredPosition = new Vector2(x, y);
        _xform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, w);
        _xform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, h);
        
        
        /*
        _xform.anchoredPosition = new Vector2(x, y);
        var width = w-x;
        var heigth = y-h;
        _xform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, width);
        _xform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, heigth);
        */
        
        // Label (class name + score)
        var name = _labels[(int)d.classIndex];
        _label.text = $"{name} {(int)(d.score * 100)}%";

        // Panel color
        var hue = d.classIndex * 0.073f % 1.0f;
        var color = Color.HSVToRGB(hue, 1, 1);
        color.a = 0.4f;
        _panel.color = color;
        _rect = _xform.rect;
        // Enable
        gameObject.SetActive(true);
    }

    public void Hide()
      => gameObject.SetActive(false);
}
