import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


def get_activation_hook(name, activations_dict):
    def hook(model, input, output):
        activations_dict[name] = output
    return hook

def train_student(student, teacher, loader, temperature=2.0, lr=1e-3, epochs=3, supp=None, device="cuda", extract=None, layer_name=None):
    optimizer = optim.Adam(student.parameters(), lr=lr)
    student.train()
    # teacher.eval()
    # For the moment the activations are not used, only the logits are used to train the student model
    # teacher_activations = {}
    # student_activations = {}
    # teacher.fc2.register_forward_hook(get_activation_hook(layer_name, teacher_activations))
    # student.fc2.register_forward_hook(get_activation_hook(layer_name, student_activations))

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        train_loss = total = correct =0.0
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            # teacher_activations.clear()
            # student_activations.clear()

            optimizer.zero_grad()

            # Get teacher and student logits
            with torch.no_grad():
                teacher_logits= teacher(images)
                    # Get activations from fc2
                # t = teacher_activations[layer_name]

            student_logits = student(images)
            # s = student_activations[layer_name]

            # Apply temperature-scaled softmax
            p_teacher = F.softmax(teacher_logits / temperature, dim=1)
            log_p_student = F.log_softmax(student_logits / temperature, dim=1)

            # Cross-entropy between teacher and student distributions
            # loss_1 = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (temperature ** 2)
            loss_1 = F.mse_loss(F.sigmoid(student_logits), F.sigmoid(teacher_logits))
            # loss_2 = F.mse_loss(s, t)
            loss =  loss_1
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            _, ber_student = extract(student, supp)
            _, ber_teacher = extract(teacher, supp)

            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"f"/{total}]", ber_student=f"{ber_student:4f}", ber_teacher=f"{ber_teacher:4f}")

    return student.state_dict()