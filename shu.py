from PIL import Image

# Mở ảnh
image = Image.open('2d18ed89303596e64d94aec0bb166773d8f54ed452576bae44cac46de6c5c324(2).png')

# Kích thước mới (width, height)
new_size = (200, 100)

# Resize ảnh
resized_image = image.resize(new_size)

# Lưu ảnh đã resize
resized_image.save('avt.png')
