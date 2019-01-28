import cv2


class Fixation:
    
    def __init__(self, index, center, radius, size=25):
        self.index = index
        self.center = center
        self.radius = radius
        self.size = size
    
    @classmethod
    def from_record(cls, record):
        index = record['FPOGID']
        center = tuple(record[['PIX_FPOGX', 'PIX_FPOGY']].values)
        radius = int(record['FPOGD'] * 10)
        return cls(index, center, radius)
    
    def update(self, record):
        self.center = tuple(record[['PIX_FPOGX', 'PIX_FPOGY']].values)
        self.radius = int(record['FPOGD'] * self.size)
    
    def draw(self, image, color, border=1, annot=True):
        # draw circle with border
        cv2.circle(image, self.center, self.radius, color, -1)
        if border:
            cv2.circle(image, self.center, self.radius+(border//2), (0, 0, 0), border)
        #tuple(map(lambda value: value + 1, self.center))
        cv2.putText(image, f'{self.index}', self.center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(image, f'{self.index}', self.center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def trace(self, image, color, fixation, border=2):
        cv2.line(image, self.center, fixation.center, color, border)
        fixation.draw(image, color, border)
        self.draw(image, color, border)

