
## 📥 Download Model Weights and Config

Before running the server, you need to download the trained model weights and config file:

🔗 **[Download model weights and config](https://drive.google.com/drive/folders/1vg3_r5jUPluVrpVkmcILy-Xnw0Uqubgk?usp=sharing)**

After downloading, place the files in the `detectron2_model/[plantName_model]` folder.  
Example folder structure:


## Running the Server
Run the app.py file to launch the server, default port is 5000 but can be changed. 

/predictPotato
/predictTomato
 
are the endpoints for respective plants.

this script will load required weights and config and place the prediction in `static/upload` as `PlantNameVisualization` Requests can be sent and images can be returned through the 2 `Clients` available - mobile app and webpage

## Tunelling Online With Ngrok

As the flask server runs on your localhost you can tunnel it online using any Tunneling Service. 

We made use of Ngrok for quick online tunnel, the sample code for tunneling it online is provided in 'run_with_ngrok.py'
Note - You may use Discord Webhook for faster acess to the link 
