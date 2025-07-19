import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# 🛠️ Configuración
model_path = "D:/Informacion/Desktop/Code/modelo_plantas_final.keras"  # o "best_modelo_plantas.h5"
test_path = "D:/Informacion/Desktop/Code/ProyectoImpacto/dataSet/Augmented Images"

# 🔄 Cargar modelo
try:
    model = load_model(model_path)
    print(f"✅ Modelo cargado exitosamente desde: {model_path}")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    exit()

# 📊 Configurar generador de datos para testing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

# 📝 Obtener nombres de las clases
class_names = list(test_generator.class_indices.keys())
print(f"Clases disponibles: {class_names}")

# 🎯 Evaluación completa del modelo
print("\n=== EVALUACIÓN COMPLETA DEL MODELO ===")
test_loss, test_acc, test_top3 = model.evaluate(test_generator, verbose=1)
print(f"Precisión en test: {test_acc:.4f}")
print(f"Top-3 Precisión: {test_top3:.4f}")
print(f"Pérdida en test: {test_loss:.4f}")

# 📈 Predicciones detalladas
print("\n=== GENERANDO PREDICCIONES ===")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
predicted_probs = np.max(predictions, axis=1)
true_classes = test_generator.classes

# 📊 Reporte de clasificación
print("\n=== REPORTE DE CLASIFICACIÓN ===")
report = classification_report(true_classes, predicted_classes, 
                             target_names=class_names, output_dict=True)
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# 📋 Crear DataFrame con resultados
results_df = pd.DataFrame({
    'filename': test_generator.filenames,
    'true_class': [class_names[i] for i in true_classes],
    'predicted_class': [class_names[i] for i in predicted_classes],
    'confidence': predicted_probs,
    'correct': true_classes == predicted_classes
})

# 📊 Análisis de confianza
print("\n=== ANÁLISIS DE CONFIANZA ===")
print(f"Confianza promedio: {predicted_probs.mean():.4f}")
print(f"Confianza mínima: {predicted_probs.min():.4f}")
print(f"Confianza máxima: {predicted_probs.max():.4f}")

# Predicciones con baja confianza
low_confidence = results_df[results_df['confidence'] < 0.5]
print(f"\nPredicciones con baja confianza (<0.5): {len(low_confidence)}")
if len(low_confidence) > 0:
    print(low_confidence[['filename', 'predicted_class', 'confidence']].head())

# 🎨 Visualizaciones
def plot_confusion_matrix(true_classes, predicted_classes, class_names):
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Normalizar para porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión Normalizada')
    plt.xlabel('Predicción')
    plt.ylabel('Realidad')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_confidence_distribution():
    plt.figure(figsize=(12, 4))
    
    # Distribución de confianza
    plt.subplot(1, 2, 1)
    plt.hist(predicted_probs, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(predicted_probs.mean(), color='red', linestyle='--', 
                label=f'Media: {predicted_probs.mean():.3f}')
    plt.xlabel('Confianza')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Confianza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confianza por clase
    plt.subplot(1, 2, 2)
    confidence_by_class = [predicted_probs[predicted_classes == i].mean() 
                          for i in range(len(class_names))]
    plt.bar(class_names, confidence_by_class, color='lightgreen', edgecolor='black')
    plt.xlabel('Clase')
    plt.ylabel('Confianza Promedio')
    plt.title('Confianza Promedio por Clase')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Generar visualizaciones
plot_confusion_matrix(true_classes, predicted_classes, class_names)
plot_confidence_distribution()

# 🔍 Función para probar imágenes individuales
def predict_single_image(image_path, model, class_names, top_k=3):
    """
    Predice la clase de una imagen individual
    """
    try:
        # Cargar y preprocessar imagen
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Hacer predicción
        predictions = model.predict(img_array, verbose=0)
        
        # Obtener top-k predicciones
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        results = []
        for i in top_indices:
            results.append({
                'class': class_names[i],
                'confidence': predictions[0][i]
            })
        
        return results
        
    except Exception as e:
        print(f"Error procesando imagen {image_path}: {e}")
        return None

# 🧪 Función para testing batch
def test_random_images(num_images=10):
    """
    Prueba imágenes aleatorias y muestra resultados
    """
    print(f"\n=== PRUEBA DE {num_images} IMÁGENES ALEATORIAS ===")
    
    # Seleccionar imágenes aleatorias
    random_indices = np.random.choice(len(test_generator.filenames), num_images, replace=False)
    
    for i, idx in enumerate(random_indices):
        filename = test_generator.filenames[idx]
        full_path = os.path.join(test_path, filename)
        
        # Obtener clase verdadera
        true_class = class_names[true_classes[idx]]
        
        # Hacer predicción
        predictions = predict_single_image(full_path, model, class_names, top_k=3)
        
        if predictions:
            print(f"\n{i+1}. {filename}")
            print(f"   Clase real: {true_class}")
            print(f"   Predicciones:")
            for j, pred in enumerate(predictions):
                print(f"     {j+1}. {pred['class']}: {pred['confidence']:.4f}")
            
            # Marcar si es correcta
            is_correct = predictions[0]['class'] == true_class
            print(f"   ✅ Correcto" if is_correct else f"   ❌ Incorrecto")

# 📊 Análisis por clase
def analyze_per_class_performance():
    """
    Analiza el rendimiento por clase
    """
    print("\n=== ANÁLISIS POR CLASE ===")
    
    class_performance = []
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        class_predictions = predicted_classes[class_mask]
        class_confidences = predicted_probs[class_mask]
        
        if len(class_predictions) > 0:
            accuracy = np.mean(class_predictions == i)
            avg_confidence = np.mean(class_confidences)
            total_samples = len(class_predictions)
            
            class_performance.append({
                'Clase': class_name,
                'Precisión': accuracy,
                'Confianza Promedio': avg_confidence,
                'Total Muestras': total_samples
            })
    
    # Crear DataFrame y mostrar
    performance_df = pd.DataFrame(class_performance)
    performance_df = performance_df.sort_values('Precisión', ascending=False)
    
    print("\nRendimiento por clase:")
    print(performance_df.to_string(index=False, float_format='%.4f'))
    
    # Visualización
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(performance_df['Clase'], performance_df['Precisión'], 
            color='lightblue', edgecolor='black')
    plt.xlabel('Clase')
    plt.ylabel('Precisión')
    plt.title('Precisión por Clase')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(performance_df['Clase'], performance_df['Confianza Promedio'], 
            color='lightcoral', edgecolor='black')
    plt.xlabel('Clase')
    plt.ylabel('Confianza Promedio')
    plt.title('Confianza Promedio por Clase')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 📤 Ejecutar análisis
analyze_per_class_performance()
test_random_images(15)

# 💾 Guardar resultados
results_df.to_csv('resultados_predicciones.csv', index=False)
print(f"\n💾 Resultados guardados en 'resultados_predicciones.csv'")

print("\n✅ Evaluación completada exitosamente!")
