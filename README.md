# Eat Blindly

  ![Alt text](src/static/EatBlindlyv.1PitchCover.jpg)


## **Eat Blindly v.1 ("Bread-Not bread" Classifier App)**

This is the beta version of an application further aimed at **curating places where to eat chef-level food** leverdaged by a model trained for analyzing and classifying visual objective features of food photos scrapped from public profiles of restaurants, bars and bakeries in Google Maps, Instagram, Pinterest, etc. 

Since each type of food has its own set of objective visual feautures when distinguishing if it is "foody-good" level or rather mediocre, the roadmap of the model training will be deployed over time by **vertical food types**, starting by the most popular food types in a main foodie cities worldwide (i.e. bread, sushi, pizza, paella, tacos, etc.). 

The visual characteristics of each vertical or food type are being carefully identified, typified and latelly embbeded to classified food images using tag-based prompts, by a pool of selected well-known gastronomers and chefs in each city.

Within each food type and potentially for the first 5-10 food types to be analyzed-deployed until the model can infere enough food classes based on its collected visual knowledge from learning about enough previous food classes, we plan to develope a first filter of the model trained at distinguishing if the image belongs or not to the food type itself, before further classifying it as "foody-good" level or not.

We plan to deploy a 1st version of the app in the city of Barcelona and with the food type "bread", taking in this case as a reference for the bread foody-level class, the objective visual features of a bread with a certain % of sourdough (90% or more). 

In order to make super easy and fast to classify and tag images of bread =>90% sourdough bread or below, we have created a tinder-like app leveredging the gamification capabilities of this UI for chefs to be able to classify and tag images as though if they were playing cards (o looking for their other half :D).

* [Eat Blindly pitch and roadmap:](https://drive.google.com/file/d/1WVHgsrFUt8kqF7vE2oddmj6azsL3wrED/view?usp=sharing)
  
  ![Alt text](src/static/EatBlindlyRoadmap.jpg)
  
* [Eat Blindy Front end mock up](https://eatblindly.my.canva.site/)

  ![Alt text](src/static/EatBlindlyMockUp.jpg)
 
 



<br>

## **Summary of steps and milestones achieved so far:**

* **Bread tagging and classification App** Flask app deployed using a temporary port url set up as public because of: 
  * A) Incompatibilities with Torch & Transformers libraries and Render that didn't allow to deploy the app using a permanent url 
  * B) High difficulty of setting up the Auth2 level process of Google Cloud Console to connect the Github Repository with the Drive Folders where the data is.
    * [Github Repository (/images folder ignored because it surpases github repo max storage with +20.000 images)](https://github.com/dianamonroe/pretrainfoodclassificationwidget)
    * CODESPACE NEEDS T BE ACTIVE FOR THE APP TO BE EXECTUED: [Temporary public bread classification and tagging app for chefs](https://5000-dianamonroe-pretrainfoo-2w8tlujr98p.ws-eu117.gitpod.io/)

  ![Alt text](src/static/EatBlindlyRoadmap.jpg)

<br>

* **Training models**
As mentioned above, before training the model to accomplish its final goal (distinguishing foody-level class bread from bread than doesn't reach this food-level class), we have initially trained the 1st layer of the model aimed at distinguishing what is bread of what is not as follows:

   * **YOLO MODEL** - 10 Training rounds epochs (+500 epochs) of the Yolo model (Yolo8n.pt and Yolo11n.pt): **[Ultralytics Yolon11.pt model](https://docs.ultralytics.com/models/yolo11/#key-features)**, pre-traiend with **[LVIS dataset](https://docs.ultralytics.com/datasets/detect/lvis/)** where bread is a class and there are + 18 not bread pastry classes.
   * YOLO was our 1st choice initially because:
     * It is the latest iteration in the Ultralytics YOLO series of real-time object detectors, cutting-edge accuracy, speed, and efficiency.
     * Counts with rnhanced Feature Extraction: improved backbone and neck architecture, which enhances feature extraction capabilities for more precise object detection and complex
       task performance.
     * It is optimized for Efficiency and Speed: refined architectural designs and optimized training pipelines, delivering faster processing speeds and maintaining an optimal balance
       between accuracy and performance.
     * Promises greater Accuracy with Fewer Parameters: higher mean Average Precision (mAP) on the COCO dataset while using 22% fewer parameters than YOLOv8, making it computationally
       efficient without compromising accuracy.
   This labelling process was simplyfied by:
     - a) refining the dataset selecting just images where the bread and not_bread object was prominent (taking 80% of the image) and located in the center (.txt files with coinciding
       file name that the .jpg file contents "1 0.5 0.5 0.8 0.8", where "1" standas for the not_bread class, 0.5 and 0.5 points to a a center object location and 0.8 0.8 the prominence
       of the object in the image)
     - b) and reducing primarly the not_bread class images to mainly pastry-related not_bread food and other not_bread no pastry-related objects similar to bread.

     After the 3rd training round of Yolo11n.pt, if seemed to perform worst in every training round despite showing outstanding numerical above 94.8% for all -overall and pre class-
     metrics.

![Alt text](static/Yolo113rdTrainingRoundMetrics.png)

   Finally and unfortunatelly, despite the great numerical metrics obtained with the validation set, Yolo model failed consistently in single image prediction test loading best 
   weights from the trainning (even also after several intents of data refinement an until a 10th training round).
   
   * **YOLO MODEL DEPLOYMENT -FAILS IN SINGLE IMAGE PREDICTION-:**
     * [Yolo11n.pt public Streamlit App](https://gourmetfoodclassifierv12.streamlit.app/)
     * It for instance classifies a lemon as bread with a 40% confindence.
     * [Repository of this 1st Yolon1.pt model converted to onnx in order to be deployed in a public STREAMLIT app](https://github.com/dianamonroe/gourmetfoodclassifierv1.2)).

![Alt text](static/YoloBadPredictionTest.png)

<br>

 * **OPEN AI CLIP (Contrastive Language-Image Pre-training) MODEL** -  [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - [Vision Transformer Base Batch 32](https://huggingface.co/docs/transformers/model_doc/vit) Since clearly Yolo11n.pt wasn't performing well in the single image test prediction and thus for our main business purpose (Restoration business discovery based on food product single image analysis and classification), we decide to switch to another LLM model and we trained OPEN AI CLIP model using 2 class prompts with quite better metrics in just the 1s training round.
 * **WHY DID WE SWITCHED TO CLIP?** 
 * **ZERO-SHORT SUPER POWERS:** REALLY GOOD at differentiating objects it hadn't been trained to classify. This is, to correctly classify croissants as "not bread" and bagels as
 "bread" even though it hasn't “seen” those specific types before. This is exactly the capability we need for the 1st level of distinction (does the image belongs to the desired food 
 class or not?)
 * Boosts image classification capabilities with its own robust specialized Transformer-based language processor trained jointly with a Vision Transformer (ViT), that learns directly
   from raw text about images, leveraging a much broader source of supervision than State-of-the-art computer vision systems trained to predict a fixed set of predetermined object
   categories.
* Benchmarked on over 30 different existing computer vision datasets (Food101, ImageNet, CIFAR10 or CIFAR100 among others).
* Have been pre-trained for predicting which caption goes with which image from SOTA image representations from scratch from a dataset of 400 million (image, text) pairs collected
  online (That is exactly the task we plan the model to do for our app, scrapping images online).
* After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks
* It matches the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained o

Although Yolo showed way better numerical metrics of than OPEN AI CLIP, **CLIP performed pretty well in single image prediciton tests thanks to it zero-shot shot classification capability", classifying quite well if an image it hadn't seen before by default (it knows if a magdalene or a lemon is not bread even thought it hadn't been trained with those images):

**CLIP METRCIS:**
* **CLIP 1st training round:**
  * Best Epoch: 50
  * Accuracy: 0.90
  * Precision: 0.95
  * Recall: 0.83

![Alt text](static/CLIP1stTrainingRoundConfusionMatrix.png)

* **CLIP 2nd trainig round:**
  * Best Epoch: 10
  * Accuracy: 0.86
  * Precision: 0.84
  * Recall: 0.89
    
![Alt text](static/CLIP2ndTrainingRoundConfusionMatrix.png)

* **CLIP DEPLOYMENT -WORKS WELL ENOUGH -0,90 ACC.- WITH SINGLE IMAGE PREDICTIONS):**
   * The **current repository stores the final version of Eat Blindly v.1 (Bread-Not bread Classifier App)**, that have been at the moment deployed using Flask in a temporary port url
     that needs to be launched from the open codespace of the repository. In this case and despite being able to convert CLIP Model to onnix to force compatibility with Streamlit, the
     onnx file is 5 times heavier than the max size allowed by Github (100MB) so it hadn't been possible to deploy the app publicly in Streamlit nor Flask.
   * [OPEN AI CLIP temporary and variable url port](https://laughing-sniffle-4jg966gw9vvp2j5vj-8000.app.github.dev/)
   * It is set to classify bread and not_bread images above a confidence level of 0,69, below what is not sure enough and thus the image wouldn't pass to the next layer (gourmet level
     bread or not) until further phases after improving performance metrics

  ![Alt text](src/static/OPENAICLIPdeploymentmodelapp.png)

  ![Alt text](src/static/CLIPbread_notbreadclassifier_correctnotbreadclassification.png)
  
  ![Alt text](src/static/CLIPbread_notbreadclassifier_notenoughconfindence.png) 

<br>
   
## **REQUEST ACCESS TO THE DATA PROCESSING AND TRAINING PROJECT REPOSITORIES:**
You can request access to a bunch of repositories mainly done in Google Colab where the different phases of the project have been developed in the following Google Drive Folders:

* 1. [Image Scrapping](https://drive.google.com/drive/folders/1w28M03pW-V66UihSKkf1pyW0xMzToP8E?usp=drive_link)
* 2. [Data Preparation, Standardization, Augmentation & Refinement](https://drive.google.com/drive/folders/1ztH7bXBSfOYk5NEBFg91tBL2P8DUVCQw?usp=sharing)
* 3. [Model Training, Evaluation & Deployment](https://drive.google.com/drive/folders/1eLGwQrhMVTj-36B4KAG4cDRBVylq2vQS?usp=drive_link)
   * [Request acces to Yolo1 training rounds & evaluation](https://drive.google.com/drive/folders/1-yJXt_jNmqBSOIssn9llxW-f9pY-Rut_?usp=drive_link)
   * [Request access to OPEN AI CLIP training rounds & evaluation](https://drive.google.com/file/d/1p9N38zwv3FTrn41g7O1fBrD4ZUZRVwMb/view?usp=drive_link)
    
