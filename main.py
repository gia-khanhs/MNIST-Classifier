import numpy as np 
import matplotlib.pyplot as plt

from utils.mlp import mlp

#=====================================================
training = False

mnistClassifier = mlp()
if training:
    mnistClassifier.train(learningRate=0.3, iterations=300)
    mnistClassifier.saveParams()

    print("Parameters saved!")

else:
    mnistClassifier.loadParams()

    accuracy = mnistClassifier.calcTestAccuracy()
    print("Parameters loaded!")
    print(f"Accuracy on test set: {accuracy}")

#=====================================================

img = np.zeros((28, 28))

drawing = {"pressed": False}

fig, ax = plt.subplots()
plt.title('Digit Recogniser (0-9)')
im = ax.imshow(img, cmap="gray", vmin=0, vmax=1) # show array as an image

ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
ax.grid(which="minor", linestyle="-", linewidth=0.5) #minor ticks => gridlines

ax.set_xticks([])
ax.set_yticks([]) # remove major ticks

def set_pixel(event):
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return

    j = int(event.xdata)  # column index
    i = int(event.ydata)  # row index

    if 0 <= i < 28 and 0 <= j < 28:
        img[i, j] = min(img[i, j] + 0.7, 1.0)

        neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        for ni, nj in neighbors:
            if 0 <= ni < 28 and 0 <= nj < 28:
                img[ni, nj] = min(img[ni, nj] + 0.4, 1.0)

        im.set_data(img)
        fig.canvas.draw_idle()

        #===================== Prediction related stuff ======================
        x = img.ravel()
        x = x.reshape((784, 1))
        y, prob = mnistClassifier.predict(x)
        
        yLabel = [str(i) + ": " + str(round(prob[i][0], 2)) for i in range(10)]
        yLabel = "\n".join(yLabel)
        plt.xlabel(f"Predicted digit: {y[0][0]}")
        plt.ylabel(yLabel, rotation=0, labelpad=20)

def on_press(event):
    drawing["pressed"] = True
    set_pixel(event)

def on_release(event):
    drawing["pressed"] = False

def on_move(event):
    if drawing["pressed"]:
        set_pixel(event)

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()