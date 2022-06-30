using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ObjectDetection
{
    public class Coco
    {
        public List<Image> images;
        public Coco(TextAsset annotations)
        {
            images = new List<Image>();
            string text = annotations.text;
            string[] entries = text.Split('\n');
            foreach (string entry in entries)
            {
                this.images.Add(new Image(entry));
            }
        }
        public class Image
        {
            public Image(string imageString)
            {
                string[] image = imageString.Split(' ');
                this.path = image[0];
                this.labels = new List<Label>();
                for (int i = 1; i < image.Length; i++)
                {
                    labels.Add(new Label(image[i]));
                }
            }
            public string path;
            public List<Label> labels;
            public class Label
            {
                public Label(string labelString)
                {
                    string[] label = labelString.Split(',');
                    this.rect = new Rect();
                    int value = 0;
                    int.TryParse(label[0], out value);
                    this.rect.xMin = value;
                    int.TryParse(label[1], out value);
                    this.rect.yMin = value;
                    int.TryParse(label[2], out value);
                    this.rect.xMax = value;
                    int.TryParse(label[3], out value);
                    this.rect.yMax = value;
                    int.TryParse(label[4], out value);
                    this.classIndex = value;
                }
                public Rect rect;
                public int classIndex;
            }
        }
    }
}
