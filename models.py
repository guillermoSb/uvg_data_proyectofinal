import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class ConvNet(nn.Module):
    def __init__(self, input_chanels, image_size):
        super(ConvNet, self).__init__()
        # Definición de las capas
        self.conv1 = nn.Conv2d(input_chanels, 32, kernel_size=3, stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normalización por lotes
        self.dropout1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Normalización por lotes
        self.dropout2 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Normalización por lotes
        self.dropout3 = nn.Dropout2d(0.25)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # Normalización por lotes
        self.dropout4 = nn.Dropout2d(0.25)

        output_size_after_conv = image_size // (2**4)

        self.fc1 = nn.Linear(256 * output_size_after_conv \
                             * output_size_after_conv, 1000)
        # self.fc1 = nn.Linear(2048, 1000)

        # num_clases debería ser el número de etiquetas
        # únicas en tu conjunto de datos
        self.fc2 = nn.Linear(1000, 1)

    def forward(self, x):
        # Capa convolucional 1 con ReLU y Max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.max_pool2d(x, 2)

        # Capa convolucional 2 con ReLU y Max pooling
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)

        # Capa convolucional 3 con ReLU y Max pooling
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = F.max_pool2d(x, 2)

        # Capa convolucional 4 con ReLU y Max pooling
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        x = F.max_pool2d(x, 2)

        # Aplanando los datos
        x = x.view(x.size(0), -1)

        # Capas completamente conectadas con ReLU y softmax para la salida
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def infer(model, image, device='cpu'):
    """
    Realiza la inferencia en una imagen usando un modelo entrenado.
    
    Args:
    model (torch.nn.Module): El modelo entrenado para inferencia.
    image_path (str): Ruta al archivo de imagen para realizar la inferencia.
    device (str): El dispositivo en el que se ejecutará la inferencia ('cpu' o 'cuda').
    
    Returns:
    La predicción del modelo.
    """
    # Asegurarse de que el modelo está en modo de evaluación
    # y en el dispositivo correcto
    model = model.to(device)
    model.eval()
    
    # Transformaciones que se deben aplicar a la imagen
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Cargar y transformar la imagen
   
    image = transform(image).unsqueeze(0)  # Añadir un batch dimension
    
    # Mover la imagen al dispositivo y realizar la inferencia
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    
    # Convertir la salida del modelo a un valor numérico
    # Remueve el batch dimension y convierte a un número
    prediction = output.squeeze().item()
    
    return prediction * 100

# Ejemplo de cómo usar la función
# Supongamos que 'model' es tu modelo entrenado y
# 'image_path' es la ruta a la imagen

# EJEMPLO DE MODEL
