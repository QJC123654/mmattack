# normalize(RGB)
def normalize(img, mean, std):
  b, c, h, w = img.shape
  n_img = ((img.permute(0, 2, 3, 1) - mean) / std).permute(0, 3, 1, 2)
  return n_img
# denormalize(RGB)
def denormalize(n_img, mean, std):
  # print(n_img.shape)
  # print(mean.shape)
  # print(std.shape)
  b, c, h, w = n_img.shape
  img = (n_img.permute(0, 2, 3, 1) * std + mean).permute(0, 3, 1, 2)
  return img