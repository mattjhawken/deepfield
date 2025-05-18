class GPIO:
    BCM = None
    OUT = None
    IN = None
    HIGH = 1
    LOW = 0

    def setmode(self, mode):
        print("GPIO setmode called")

    def setup(self, pin, mode):
        print(f"GPIO setup called for pin {pin} mode {mode}")

    def output(self, pin, value):
        print(f"GPIO output set pin {pin} to {value}")

    def input(self, pin):
        return 0

    def cleanup(self):
        print("GPIO cleanup called")


GPIO = GPIO()
