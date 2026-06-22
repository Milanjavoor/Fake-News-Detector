from IPython.core import history
from os import truncate
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU , Embedding , Dense

#-------------- Load dataset --------------------------------------------------------------------------------------------------------
import os

print(os.path.getsize("/content/Fake.csv"))

fake=pd.read_csv(r"/content/Fake.csv", engine='python')
true=pd.read_csv(r"/content/True.csv", engine='python')
print(fake.head())
print(true.head())

fake["label"]=0
true["label"]=1
df=pd.concat([fake,true],ignore_index=True)
df["content"]=df["title"]+" "+df["text"]
x=df["content"]
y=df["label"]

#---------------- Tokenization -------------------------------------------------------------------------------------------------

vocabsize=10000
maxtok=300
tokenizer=Tokenizer(
    num_words=vocabsize,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(x)
sequences=tokenizer.texts_to_sequences(x)
padded=pad_sequences(
    sequences,
    maxlen=maxtok,
    padding="post",
    truncating="post"
)

#------------------- Splitting data ---------------------------------------------------------------------------------------------------

x_train,x_test,y_train,y_test=train_test_split(padded,y,test_size=0.2,random_state=42)

 # --------------- Building the GRU Model ------------------------------------------------------------------------------
model=Sequential([Embedding(vocabsize,128),
                 GRU(64),
                 Dense(32,activation="relu"),
                 Dense(1,activation="sigmoid")])
model.compile(
     optimizer="adam",
     loss="binary_crossentropy",
     metrics=["accuracy"]
 )
# -------------------- Training --------------------------------------------------------------------------------------------------------

earlystop=EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history=model.fit(x_train,y_train,
                  epochs=33,
                  batch_size=64,
                  validation_data=(x_test,y_test),
                  callbacks=[earlystop]
                  )
#------------------- Saving the model --------------------------------------------------------------------------------------------------------------------------------

model.save("fake_newsgru.keras")
with open("tokenizer.pkl","wb") as f:
  pickle.dump(tokenizer,f)


print("Training complete")
