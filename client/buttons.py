import signal
import RPi.GPIO as GPIO

# Gpio pins for each button (from top to bottom)
BUTTONS = [5, 6, 16, 24]

# These correspond to buttons A, B, C and D respectively
LABELS = ['A', 'B', 'C', 'D']

# Set up RPi.GPIO with the "BCM" numbering scheme
GPIO.setmode(GPIO.BCM)

# Buttons connect to ground when pressed, so we should set them up
# with a "PULL UP", which weakly pulls the input signal to 3.3V.
GPIO.setup(BUTTONS, GPIO.IN, pull_up_down=GPIO.PUD_UP)


# "function" will be called every time a button is pressed
# It receives one argument: the associated input pin.
def set_button_function(button_label, function):
    """
    "function" will be called every time a button is pressed
    :param button_label: upper case letter ['A', 'B', 'C', 'D']
    :param function: function that should be run on button press
    :return:
    """
    if button_label in LABELS:
        pin = BUTTONS[LABELS.index(button_label)]
        GPIO.add_event_detect(pin, GPIO.FALLING, function, bouncetime=250)
    else:
        raise Exception(f"{button_label} is not a valid button label")


def wait_forever_for_button_presses():
    signal.pause()
