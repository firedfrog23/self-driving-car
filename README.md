# Self-Driving Car - Semantic Segmentation Project

## 1. Business Understanding
Kemacetan dan kecelakaan lalu lintas menjadi permasalahan utama yang dapat dikurangi dengan teknologi self-driving car. Dalam proyek ini, dikembangkan sistem berbasis AI yang mampu melakukan segmentasi objek dalam lingkungan perkotaan menggunakan dataset Cityscapes.

Tujuan utama proyek ini:
- Mengembangkan model deep learning untuk segmentasi objek.
- Menggunakan arsitektur **FCN-8s** dan **U-Net**.
- Meningkatkan efisiensi transportasi dan keselamatan jalan raya.
- Mengintegrasikan model ke dalam sistem real-time untuk self-driving car.
- Mengoptimalkan performa model dengan teknik fine-tuning.

---

## 2. Data Understanding
Dataset yang digunakan adalah **Cityscapes**, yang terdiri dari gambar dengan 12 kelas segmentasi. Dataset ini diproses menggunakan **torchvision.transforms** untuk augmentasi dan normalisasi data guna meningkatkan kinerja model.

Beberapa langkah eksplorasi data:
- Visualisasi sampel gambar dan mask segmentasi.
- Penggunaan GPU untuk percepatan pelatihan.
- Pembagian dataset menjadi training dan testing.
- Analisis distribusi kelas dalam dataset untuk memastikan tidak ada ketidakseimbangan data yang signifikan.
- Preprocessing tambahan seperti data augmentation (flipping, rotation, dan color jittering) untuk meningkatkan generalisasi model.

---

## 3. Algorithm Definition
### **U-Net**
- Menggunakan struktur **encoder-decoder** dengan **skip connections** untuk mempertahankan informasi spasial.
- Menggunakan **Double Convolution Block** dengan batch normalization dan ReLU activation.
- Sangat cocok untuk segmentasi objek dalam berbagai kondisi.
- Dapat menangani gambar dengan resolusi berbeda tanpa kehilangan informasi penting.
- Efisien dalam memproses gambar dengan berbagai tingkat kompleksitas.

### **FCN (Fully Convolutional Network)**
- Mengganti lapisan **fully connected** dengan lapisan **konvolusi**.
- Memanfaatkan teknik **upsampling/deconvolution** untuk meningkatkan resolusi segmentasi.
- FCN-8s digunakan untuk mempertahankan detail segmentasi lebih baik.
- Mampu mengadaptasi ukuran gambar input yang bervariasi.
- Menghasilkan prediksi segmentasi dengan tingkat akurasi tinggi.

---

## 4. Training Process
1. **Persiapan Dataset**: Dataset Cityscapes dikustomisasi dengan **CityscapesDataset class**.
2. **Model dan Optimizer**: Implementasi dengan PyTorch, menggunakan Adam optimizer dan **Cross-Entropy Loss**.
3. **Forward & Backward Propagation**: Training dengan backpropagation untuk memperbarui bobot model.
4. **Evaluasi Performa**:
   - Menggunakan **Intersection over Union (IoU)** dan **Dice Coefficient**.
   - Logging **training loss**, **validation loss**, dan **accuracy** setiap epoch.
   - Menyimpan model terbaik berdasarkan performa di validation set.
5. **Hyperparameter Tuning**:
   - Learning rate: 0.0001 - 0.001
   - Batch size: 8 atau 16
   - Epochs: 50
   - Dropout dan regularization digunakan untuk mengurangi overfitting.

---

## 5. Results
- **Average IoU pada Test Set**: 58.93%
- **Training Loss**: 0.1497 (epoch 50/50)
- **Validation Loss**: 0.2641
- **Validation IoU**: 60.25%
- **Overfitting ringan**, namun masih bisa diperbaiki dengan tuning lebih lanjut.
- Visualisasi hasil segmentasi menunjukkan bahwa model mampu mendeteksi dan membedakan objek dengan baik dalam kondisi normal.

---

## 6. Deployment
1. **Menyimpan Model**: Model terbaik disimpan dalam format `.pt` untuk keperluan deployment.
2. **Integrasi dengan API**: Model dapat dieksekusi dalam lingkungan cloud menggunakan **Flask/FastAPI**.
3. **Inference Pipeline**:
   - Menggunakan OpenCV untuk preprocessing gambar input.
   - Model melakukan prediksi segmentasi dan mengembalikan hasil dalam bentuk mask overlay.
   - Hasil segmentasi divisualisasikan dan disesuaikan agar mudah dipahami oleh sistem self-driving.
4. **Optimasi untuk Real-time Processing**:
   - Model dioptimalkan menggunakan TensorRT untuk inference yang lebih cepat.
   - Implementasi pada hardware GPU dengan CUDA untuk performa tinggi.
   - Penggunaan model yang lebih ringan seperti MobileNet untuk deployment di perangkat edge.

---

## 7. Conclusion
Proyek ini berhasil membangun model segmentasi objek menggunakan deep learning, khususnya dengan arsitektur **U-Net** dan **FCN-8s**. Model ini memiliki potensi untuk diterapkan dalam sistem persepsi kendaraan otonom.

Kedepannya, beberapa peningkatan yang dapat dilakukan:
- **Fine-tuning lebih lanjut** dengan dataset yang lebih besar dan kompleks.
- **Menggunakan model yang lebih ringan** untuk implementasi di perangkat dengan keterbatasan sumber daya.
- **Eksperimen dengan teknik transfer learning** untuk meningkatkan akurasi segmentasi.

---

## 8. How to Run
1. Clone repository ini:
   ```bash
   git clone https://github.com/username/self-driving-car-segmentation.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan notebook:
   ```bash
   jupyter notebook project3_self_driving_car.ipynb
   ```
4. Latih model:
   ```python
   python train.py
   ```
5. Jalankan API untuk inference:
   ```bash
   python app.py
   ```
6. Kirim request ke API:
   ```bash
   curl -X POST -F "file=@test_image.png" http://127.0.0.1:5000/predict
   ```

---

## 9. References
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [FCN Paper](https://arxiv.org/abs/1411.4038)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorRT Optimization](https://developer.nvidia.com/tensorrt)

---

Terima Kasih!

_Created as part of the Self-Driving Car AI Project._
