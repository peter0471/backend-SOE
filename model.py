from io import BytesIO
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:
    def __init__(self):
        self.model = tf.keras.models.load_model(
            r"ResNet101_pretrained_weights.h5")
        self.labels = ["retracted tm:", "safe csom:",
                       "normal:", "unsafe csom:", "acute otitis media:"]
        self.extra_images = []

    def update_files(self):
        fd = {"train":[]}
        with open('complicated.db','r') as f:
            fd = json.load(f)

        with open('complicated.db','w') as f:
            for d in self.extra_images:
                fd["train"].append({"b64":d.b64,"lab":d.label})
            json.dump(fd,f,indent=4)

        self.extra_images.clear()

    def weekly_update(self):
        self.update_files()

    def predict(self, file):
        try:
            badstr = "data:image/jpeg;base64,"
            if (file.startswith(badstr)):
                l = len(badstr)
                file = file[l:]
            with open("inp.jpg", "wb") as f:
                f.write(base64.b64decode(file))
            image = np.expand_dims(np.asarray(
            (Image.open(BytesIO(
                base64.b64decode(file)
                )))
            .resize((224, 224)))[..., :3], 0)
            cat = list(self.model.predict(image))[0]
            for i in range(len(cat)):
                print(cat[i])
                print(self.labels[i], cat[i]*100, "%")
            return {"Class": self.labels[np.argmax(cat)]}
        except:
            return -1
    
    def update_dataset(self,data):
        self.extra_images.append(data)
        self.weekly_update()
        print(self.extra_images)
        
