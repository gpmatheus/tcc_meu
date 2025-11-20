import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# --- Configurações ---
hdf5_file = "data/TCIR-ATLN_EPAC_WPAC.h5"   # coloque o caminho do arquivo aqui
dataset_key = "matrix"         # nome do dataset dentro do h5 (ex: "images")
chunck_size = 100

# --- Carregar dataset ---
with h5py.File(hdf5_file, "r") as f:
    print(f"Total de imagens: {f[dataset_key].shape[0]}")
    global images
    images = f[dataset_key][0:chunck_size]
    print(f"Imagens carregadas: {images.shape[0]}")

# --- Estado da navegação ---
index = {"i": 0, "chunck": 0, "channel": 0}

# --- Função para mostrar imagem ---
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)  # espaço pros botões
img_plot = ax.imshow(images[index["i"]], cmap="gray")
ax.set_title(f"Imagem {index['i']+1}/{len(images)}")
ax.axis("off")

def update_image():
    img_plot.set_data(images[index["i"], :, :, index["channel"]])
    ax.set_title(f"Imagem {index['i']+1}/{len(images)}")
    fig.canvas.draw_idle()

# --- Callbacks dos botões ---
def next_image(event):
    global images
    if index["i"] < len(images) - 1:
        index["i"] += 1
        update_image()
    else:
        print("Carregando mais imagens...")
        with h5py.File(hdf5_file, "r") as f:
            start = (index["chunck"] + 1) * chunck_size
            end = start + chunck_size
            if start < f[dataset_key].shape[0]:
                new_images = f[dataset_key][start:end]
                images = new_images
                index["i"] = 0
                index["chunck"] += 1
                update_image()
            else:
                print("Não há mais imagens para carregar.")

def prev_image(event):
    if index["i"] > 0:
        index["i"] -= 1
        update_image()
    else:
        print("Carregando imagens anteriores...")
        if index["chunck"] > 0:
            with h5py.File(hdf5_file, "r") as f:
                start = (index["chunck"] - 1) * chunck_size
                end = start + chunck_size
                new_images = f[dataset_key][start:end]
                global images
                images = new_images
                index["i"] = len(images) - 1
                index["chunck"] -= 1
                update_image()
        else:
            print("Você já está na primeira imagem.")

def set_channel(event, channel):
    index["channel"] = channel
    update_image()

# --- Criar botões ---
axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
ax0 = plt.axes([0.2, 0.05, 0.1, 0.075])
ax1 = plt.axes([0.3, 0.05, 0.1, 0.075])
ax2 = plt.axes([0.4, 0.05, 0.1, 0.075])
ax3 = plt.axes([0.5, 0.05, 0.1, 0.075])
axnext = plt.axes([0.6, 0.05, 0.1, 0.075])
bprev = Button(axprev, "⟵ Esquerda")
bt0 = Button(ax0, "0")
bt1 = Button(ax1, "1")
bt2 = Button(ax2, "2")
bt3 = Button(ax3, "3")
bnext = Button(axnext, "Direita ⟶")

bprev.on_clicked(prev_image)
bt0.on_clicked(lambda event: set_channel(event, 0))
bt1.on_clicked(lambda event: set_channel(event, 1))
bt2.on_clicked(lambda event: set_channel(event, 2))
bt3.on_clicked(lambda event: set_channel(event, 3))
bnext.on_clicked(next_image)

plt.show()
