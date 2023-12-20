import numpy as np

def loss_to_rgb(loss, limit):
    loss = np.clip(loss, -limit, limit)
    return np.asarray([(loss / limit) * 0.5 + 0.5, 1-((loss / limit) * 0.5 + 0.5), np.zeros_like(loss)]).transpose(1,2,0)

if __name__ == '__main__':
    #create array filled with random flaots
    loss = np.random.rand(100, 100)*5.0-2
    print(loss.shape)
    rgb = loss_to_rgb(loss, 1)
    print(rgb.shape)

    import matplotlib.pyplot as plt


    plt.scatter(0,1,color = (0,0,1,1))
    plt.show()

    plt.imshow(rgb)
    plt.show()

def show_img(dataloader):
  for i, data in enumerate(dataloader):
    inputs, labels = data
    plt.imshow(inputs[0].permute(1,2,0))
    plt.show()
    print(inputs[0])
    print(labels[0])
    if i == 2:
      break