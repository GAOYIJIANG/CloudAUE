# 1.CloudAUE
The code of Satellite image cloud and forest fire automatic annotator with uncerntainty estimation
# 2.Run project
In the code, two pieces of code are provided: a labeled dataset as the validation dataset, a code document that provides specific parameters, and a code document that uses uncertainty evaluation as the evaluation criterion. Some remote sensing images of HRC are also provided as experimental images. The Landsat8 dataset is too large to upload and can be obtained from public datasets.

# 2.1 AnnotationCompare

In this code, the image can be determined by the following code:

```matlab
num = 1;
filename = strcat('fire',num2str(num),'.png');
```

By using the above code, the image can be accurately located, and then clicking Run in Matlab will pop up the selected image. Next, first select the convex hull of the cloud or fire in the pop-up image. The selection process involves using the mouse to click on the contour of the cloud or fire in sequence. After accurately selecting the cloud or fire, press the enter key to confirm. Then, use the mouse to click on the background part in sequence to select the same non cloud or non fire part. Press the enter key again to confirm. After a short period of time, the recognition image result and corresponding parameters will pop up.

# 2.2 Annotation

In this code, the same code is used to locate the position of the image and then annotate the image as described in the above document. However, this code is a module that annotates unlabeled datasets, and the obtained result represents the confidence score. When the obtained result is greater than 80% or the number of annotations reaches 3, the annotation ends.