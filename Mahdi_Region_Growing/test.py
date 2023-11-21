class RegionGrowing:
    def __init__(self, image):
        self.image = image
        self.rows, self.cols = image.shape[:2]
        self.visited = np.zeros_like(image)
        self.result = np.zeros_like(image)

    def region_grow(self, seed):
        stack = [seed]
        seed_value = self.image[seed]

        while stack:
            current_point = stack.pop()
            x, y = current_point

            if 0 <= x < self.rows and 0 <= y < self.cols and not self.visited[x, y]:
                if np.abs(self.image[x, y] - seed_value) < 30:  # You can adjust this threshold
                    self.visited[x, y] = 1
                    self.result[x, y] = self.image[x, y]

                    stack.append((x + 1, y))
                    stack.append((x - 1, y))
                    stack.append((x, y + 1))
                    stack.append((x, y - 1))

    def select_seed_point(self, seed):
        self.region_grow(seed)

    def save_result(self, output_path):
        cv2.imwrite(output_path, self.result)













def region_growing_manual_selecting_seed(image_path):

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create an instance of the RegionGrowing class
    region_growing = RegionGrowing(original_image.copy())

    # Specify the seed point (you can set it manually)
    seed_point = (100, 100)

    # Perform region growing with the specified seed point
    region_growing.select_seed_point(seed_point)

    # # Save the result to a file
    output_path = "result_image.jpg"
    region_growing.save_result(output_path)

    # Display the original and result images using IPython.display
    display(Image(filename=image_path, width=300))
    display(Image(filename=output_path, width=300))