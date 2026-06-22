import pickle 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model("fake_newsgru.keras")
with open("tokenizer.pkl","rb") as f:
  tokenizer=pickle.load(f)
maxlength=300
news=input("Enter the news article :\n")

sequence=tokenizer.texts_to_sequences([news])
padded=pad_sequences(
    sequence,
    maxlen=maxlength,
    padding="post",
    truncating="post"
)

prediction=model.predict(padded)[0][0]

print("confidence:",prediction)
if prediction<0.5:
  print("Fake news detected")
  img=mpimg.imread("Fakenews.jpg")
  plt.imshow(img)
  plt.axis("off")
  plt.show()

else:
  print("The news article is true")

  imga=mpimg.imread("true.jfif")
  plt.imshow(imga)
  plt.axis("off")
  plt.show()
