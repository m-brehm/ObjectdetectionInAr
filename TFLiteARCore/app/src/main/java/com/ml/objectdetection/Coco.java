package com.ml.objectdetection;

import java.util.ArrayList;
import java.util.List;

import android.content.res.AssetManager;
import android.graphics.RectF;

import com.common.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Coco
{
    private static final Logger LOGGER = new Logger();
    public List<Image> images;
    public Coco(final AssetManager assetManager,final String filename)throws IOException
    {
        images = new ArrayList<Image>();

        //String actualFilename = filename.split("file:///android_asset/")[1];
        InputStream labelsInput = assetManager.open(filename);
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
        String line;
        while ((line = br.readLine()) != null) {
            LOGGER.w(line);
            this.images.add(new Image(line));
        }
        br.close();
    }
    public class Image
    {
        public Image(String imageString)
        {
            String[] image = imageString.split(" ");
            this.path = image[0];
            this.labels = new ArrayList<Label>();
            for (int i = 1; i < image.length; i++)
            {
                labels.add(new Label(image[i]));
            }
        }
        public String path;
        public List<Label> labels;
        public class Label
        {
            public Label(String labelString)
            {
                String[] label = labelString.split(",");
                this.rect = new RectF();
                this.rect.left = Integer.parseInt(label[0]);
                this.rect.top = Integer.parseInt(label[1]);
                this.rect.right = Integer.parseInt(label[2]);
                this.rect.bottom = Integer.parseInt(label[3]);
                this.classIndex = Integer.parseInt(label[4]);
            }
            public RectF rect;
            public int classIndex;
            public boolean detected;
        }
    }
}
