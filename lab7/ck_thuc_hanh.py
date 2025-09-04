# Import các thư viện cần thiết
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Bước 1: Load dữ liệu và trả về input và target output
def load_data():
    # Tải dữ liệu CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Chuẩn hóa dữ liệu về khoảng [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Chuyển y_train, y_test thành one-hot vector cho đầu ra của mô hình
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Trả về x_train, y_train (input và output cho huấn luyện), x_test, y_test cho kiểm thử
    return x_train, y_train, x_test, y_test

# Bước 2: Khai báo mô hình bằng ảnh
def create_model():
    # Khởi tạo mô hình CNN
    model = Sequential()
    
    # Lớp tích chập và pooling đầu tiên
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    # Lớp tích chập và pooling thứ hai
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Lớp tích chập và pooling thứ ba
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Lớp làm phẳng và lớp fully-connected
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    
    # Lớp đầu ra với softmax cho 10 lớp
    model.add(Dense(10, activation='softmax'))
    
    # Compile mô hình
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Thực hiện các bước
x_train, y_train, x_test, y_test = load_data()
model = create_model()

# Bước 3: Huấn luyện mô hình với dữ liệu train
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Bước 4: Đánh giá mô hình trên tập kiểm thử
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Độ chính xác trên tập kiểm thử:", test_accuracy)
