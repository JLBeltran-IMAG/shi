import smaract.ctl as ctl
import sys
from pathlib import Path


def assert_lib_compatibility():
    """
    Verifies compatibility between the Python API and the loaded shared library version.

    This function checks that the major version numbers of the Python API and the 
    loaded shared library are the same to prevent errors caused by version incompatibilities.
    If the major version numbers do not match, a RuntimeError is raised.

    Parameters:
    -----------
    None

    Raises:
    -------
    RuntimeError
        If the major version number of the Python API does not match that of the 
        loaded shared library, indicating a version incompatibility.

    Notes:
    ------
    The function retrieves the API version using `ctl.api_version` and the shared 
    library version using `ctl.GetFullVersionString()`. It then compares the major 
    version numbers to ensure they are consistent.
    """
    vapi = ctl.api_version
    vlib = [int(i) for i in ctl.GetFullVersionString().split('.')]
    if vapi[0] != vlib[0]:
        raise RuntimeError("Incompatible SmarActCTL python api and library version.")


def find_device():
    """
    Searches for connected devices using the SmarActCTL library.

    This function attempts to locate devices connected to the system using the 
    `ctl.FindDevices` method. If a device is found, it returns the device information.
    If no devices are found or if an error occurs during the search, the function 
    prints an error message and exits the program.

    Parameters:
    -----------
    None

    Returns:
    --------
    str
        A buffer containing the information of the connected device(s) if found.

    Raises:
    -------
    SystemExit
        Exits the program with status code 1 if no devices are found or if an error 
        occurs during the device search.

    Notes:
    ------
    - The function uses `ctl.FindDevices("", 1024)` to search for devices. 
    - If no devices are detected (`len(buffer) == 0`), the program prompts the user 
      and then exits with an error status.
    - Any exception during the execution is caught, and an error message is displayed 
      before exiting the program.

    Example:
    --------
    device_info = find_device()
    if device_info:
        # Proceed with further operations using the connected device.
    """
    try:
        buffer = ctl.FindDevices("", 1024)
        if len(buffer) != 0:
            print("Device connected")
            return buffer
        else:
            print("MCS2 no devices found.")
            input()
            exit(1)
    except:
        print("MCS2 failed to find devices. Exit.")
        input()
        exit(1)


def calibrate(d_handle, channel):
    """
    Calibrates a specified channel of the connected device using the SmarActCTL library.

    This function sets the calibration options for the given channel and then performs 
    the calibration process. The calibration is done using the device handle and 
    channel number provided as arguments.

    Parameters:
    -----------
    d_handle : int
        The device handle representing the connected device, used to access and control 
        its properties and functions.
        
    channel : int
        The specific channel of the device to be calibrated.

    Returns:
    --------
    None

    Notes:
    ------
    - The function sets the calibration options to `0` using `ctl.SetProperty_i32`, which 
      may correspond to a default or specific configuration required for calibration.
    - The actual calibration is then performed using `ctl.Calibrate`.
    - It is assumed that `d_handle` and `channel` are valid and that the device is 
      correctly connected.

    Example:
    --------
    calibrate(device_handle, 1)
    """
    ctl.SetProperty_i32(d_handle, channel, ctl.PropertyKey.CALIBRATION_OPTIONS, 0)
    ctl.Calibrate(d_handle, channel)


def waitForEvent(d_handle):
    """
    Waits for and handles events from the connected device using the SmarActCTL library.

    This function listens for events from the device associated with the given device handle.
    It specifically checks for movement-related events and returns appropriate messages or 
    error codes based on the event type and status.

    Parameters:
    -----------
    d_handle : int
        The device handle representing the connected device, used to listen for events.

    Returns:
    --------
    str or int
        - If the event is `MOVEMENT_FINISHED` with no errors (`ctl.ErrorCode.NONE`), the function 
          returns `ctl.ErrorCode.NONE`.
        - If an error occurs during the movement, it returns a formatted string indicating the 
          channel, error code, and error description.
        - If any other event is received, it returns a string describing the event.
        - If an exception occurs (e.g., timeout or other error), it returns an appropriate error message.

    Exceptions:
    -----------
    ctl.Error
        If an error occurs during the event wait process, such as a timeout or other error 
        reported by the SmarActCTL library.

    Notes:
    ------
    - The function uses `ctl.WaitForEvent` with `ctl.INFINITE` to wait indefinitely until 
      an event is received.
    - It specifically checks for the event type `MOVEMENT_FINISHED` and handles the case 
      where no error occurs or returns a detailed message if an error code is detected.
    - The function also handles timeout errors or other exceptions raised by the `ctl.Error` class.

    Example:
    --------
    result = waitForEvent(device_handle)
    if result == ctl.ErrorCode.NONE:
        # Proceed knowing the movement completed successfully.
    else:
        print(result)  # Handle or log the error information.
    """
    try:
        event = ctl.WaitForEvent(d_handle, ctl.INFINITE)

        if event.type == ctl.Event.MOVEMENT_FINISHED:

            if event.i32 == ctl.ErrorCode.NONE:
                return ctl.ErrorCode.NONE

            else:
                return "MCS2 movement finished, channel: {}, error: 0x{:04X} ({}) ".format(event.idx, event.i32, ctl.GetResultInfo(event.i32))

        else: return "MCS2 received event: {}".format(ctl.GetEventInfo(event))

    except ctl.Error as e:
        if e.code == ctl.ErrorCode.TIMEOUT: return "MCS2 wait for event timed out after {} ms".format(ctl.INFINITE)
        else: return "MCS2 {}".format(ctl.GetResultInfo(e.code))


def stop(d_handle, channel):
    """
    Stops the movement of a specified channel on the connected device using the SmarActCTL library.

    This function issues a stop command to the specified channel of the device associated with 
    the provided device handle. It also prints a message indicating which channel is being stopped.

    Parameters:
    -----------
    d_handle : int
        The device handle representing the connected device, used to send the stop command.
        
    channel : int
        The specific channel of the device that is to be stopped.

    Returns:
    --------
    None

    Notes:
    ------
    - The function uses `ctl.Stop` to stop the movement on the specified channel.
    - A message is printed to indicate the channel being stopped, which can be helpful for 
      logging or debugging purposes.

    Example:
    --------
    stop(device_handle, 1)
    """
    print("MCS2 stop channel: {}.".format(channel))
    ctl.Stop(d_handle, channel)


def move(d_handle, channel, move_mode, move_value):
    """
    Executes movement on a specified channel of the connected device using the SmarActCTL library.

    This function sets movement parameters such as velocity, acceleration, or step frequency and amplitude
    based on the specified movement mode. It then initiates the movement with the given value. The function 
    supports three movement modes: absolute movement, relative movement, and stepping.

    Parameters:
    -----------
    d_handle : int
        The device handle representing the connected device, used to control its channels.
        
    channel : int
        The specific channel of the device where the movement will be executed.
        
    move_mode : int
        The mode of movement. It can be one of the following:
        - `ctl.MoveMode.CL_ABSOLUTE`: Move to an absolute position.
        - `ctl.MoveMode.CL_RELATIVE`: Move relative to the current position.
        - `ctl.MoveMode.STEP`: Perform a stepping movement.

    move_value : int
        The value specifying the movement. It represents:
        - Absolute position (in pm) when `move_mode` is `CL_ABSOLUTE`.
        - Relative distance (in pm) when `move_mode` is `CL_RELATIVE`.
        - Number of steps when `move_mode` is `STEP`.

    Returns:
    --------
    None

    Notes:
    ------
    - For `CL_ABSOLUTE` mode, velocity and acceleration are set to high values, and the absolute position 
      is specified. The value must be within the valid range (e.g., ±10000000 pm for Piezo Scanner channels).
    - For `CL_RELATIVE` mode, velocity and acceleration are set differently, and the relative movement 
      value and direction are specified.
    - For `STEP` mode, step frequency and amplitude are set. The step amplitude must be within the valid range 
      (0 to 65535), corresponding to 0 to 100V piezo voltage. A lower amplitude results in a smaller step width.
    - The function does not wait for the movement to complete. It initiates the movement and returns immediately. 
      You can monitor the channel state using `ChannelState.ACTIVELY_MOVING` and `ChannelState.CLOSED_LOOP_ACTIVE` 
      flags to determine when the movement ends.

    Example:
    --------
    # Move to an absolute position
    move(device_handle, 1, ctl.MoveMode.CL_ABSOLUTE, 1000000)

    # Move to a relative position
    move(device_handle, 1, ctl.MoveMode.CL_RELATIVE, 500000)

    # Perform a step movement
    move(device_handle, 1, ctl.MoveMode.STEP, 100)
    """
    # Set move velocity [in pm/s].
    # Set move acceleration [in pm/s2].

    if move_mode == ctl.MoveMode.CL_ABSOLUTE:
        ctl.SetProperty_i64(d_handle, channel, ctl.PropertyKey.MOVE_VELOCITY, 1_000_000_000)
        ctl.SetProperty_i64(d_handle, channel, ctl.PropertyKey.MOVE_ACCELERATION, 1_000_000_000)
        # Specify absolute position [in pm].
        # (For Piezo Scanner channels adjust to valid value within move range, e.g. +-10000000.)
        # move_value = 1000000000

    elif move_mode == ctl.MoveMode.CL_RELATIVE:
        ctl.SetProperty_i64(d_handle, channel, ctl.PropertyKey.MOVE_VELOCITY, 500_000_000)
        ctl.SetProperty_i64(d_handle, channel, ctl.PropertyKey.MOVE_ACCELERATION, 10_000_000_000)
        # Specify relative position distance [in pm] and direction.
        # (For Piezo Scanner channels adjust to valid value within move range, e.g. 10000000.)
        # move_value = 500000000

    elif move_mode == ctl.MoveMode.STEP:
        ctl.SetProperty_i32(d_handle, channel, ctl.PropertyKey.STEP_FREQUENCY, 1_000)
        ctl.SetProperty_i32(d_handle, channel, ctl.PropertyKey.STEP_AMPLITUDE, 65_535)
        # Set maximum step amplitude [in dac increments].
        # valid range: 0 to 65535 corresponding to 0 to 100V piezo voltage
        # Lower amplitude values result in smaller step width.
        # Specify the number of steps to perform and the direction.
        # move_value = 500

    # Start actual movement.
    ctl.Move(d_handle, channel, move_value, 0)
    # Note that the function call returns immediately, without waiting for the movement to complete.
    # The "ChannelState.ACTIVELY_MOVING" (and "ChannelState.CLOSED_LOOP_ACTIVE") flag in the channel state
    # can be monitored to determine the end of the movement.


def positions_storing(position, path, interval):
    """
    Stores a position value in a text file within a specified directory.

    This function takes a position value and stores it in a file named `positions.txt` 
    located in the specified directory. If the directory exists, the function appends 
    the position to the file; if the directory does not exist, it prints an error message.

    Parameters:
    -----------
    position : int or float
        The position value to be stored in the file.
        
    path : str
        The path to the directory where the `positions.txt` file is (or should be) located.

    Returns:
    --------
    None

    Notes:
    ------
    - If the specified directory exists, the function appends the position value to 
      `positions.txt` in append mode ("at").
    - If the directory does not exist, an error message is printed indicating that the 
      file cannot be created.
    - The function does not handle any other exceptions that may occur during file operations 
      (e.g., permissions issues).

    Example:
    --------
    positions_storing(1000, "/path/to/directory")
    """
    directory = Path(path)
    filename = Path("{}/positions{}.txt".format(directory.name, interval))

    if directory.exists():
        file = open("{}/{}".format(directory, filename.name), "at")
        file.write(str(position) + "\n")
        file.close()

    else:
        print("There is no directory. Not possible to create a file with positions")


def last_position(path):
    directory = Path(path)
    position_files = list(directory.glob("*.txt"))
    position_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    last_file = position_files[0]

    with open(last_file, "r") as file:
        for i in file.readlines():
            last_angle = i.strip()

    return last_angle



# ------------------------------------------------------------------
#            ESTA FUNCION ESTA REPETIDA EN CORRECTION_CT
# ------------------------------------------------------------------
def last_index(path):
    directory = Path(path)
    image_files = list(directory.glob("*.tif"))
    image_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    last_image_index = int(alphanumeric_chars_in_filename(image_files[0].stem))

    return last_image_index

def alphanumeric_chars_in_filename(filename):
    return "".join(list(filter(str.isdigit, filename)))
# ------------------------------------------------------------------
# 
# ------------------------------------------------------------------

# =====================================================================================================================
def starting_motor(moveMode):
    """
    Initializes the motor and returns the device handle, channel, and movement mode.

    This function initializes the motor by checking library compatibility, locating the device, 
    opening the device connection, and enabling the amplifier for the specified channel. It returns 
    the device handle, channel, and movement mode based on the user-specified `moveMode` parameter.

    Parameters:
    -----------
    moveMode : str
        Specifies the movement mode for the motor. It can take one of the following values:
        - `"relative"`: Sets the movement mode to `ctl.MoveMode.CL_RELATIVE`.
        - `"absolute"`: Sets the movement mode to `ctl.MoveMode.CL_ABSOLUTE`.
        - `"step"`: Sets the movement mode to `ctl.MoveMode.STEP`.
        Any other value will not return a mode and will simply pass.

    Returns:
    --------
    tuple
        A tuple containing:
        - `d_handle` (int): The device handle representing the connected device.
        - `channel` (int): The channel that is being used (default is 0).
        - `move_mode` (int): The corresponding move mode as per the input `moveMode`.

    Raises:
    -------
    ctl.Error
        If an error occurs during the device connection or operation. Prints detailed error information 
        and the line number where the error occurred.
        
    Exception
        If any unexpected error occurs, it prints the error message and raises the exception.

    Notes:
    ------
    - The function first ensures compatibility between the API and the shared library.
    - It locates the device using the `find_device` function and opens a connection with the handle returned.
    - It enables the amplifier for channel `0` using the property `ctl.PropertyKey.AMPLIFIER_ENABLED`.
    - Depending on the `moveMode` value provided, it returns the appropriate movement mode. If the value is 
      not recognized, the function simply passes without returning a mode.

    Example:
    --------
    handle, channel, mode = starting_motor("absolute")
    """
    assert_lib_compatibility()

    d_handle = 0
    try:
        locator = find_device()
        d_handle = ctl.Open(locator, "")
        channel = 0
        ctl.SetProperty_i32(d_handle, channel, ctl.PropertyKey.AMPLIFIER_ENABLED, ctl.ENABLED)

        if moveMode == "relative": return d_handle, channel, ctl.MoveMode.CL_RELATIVE
        elif moveMode == "absolute": return d_handle, channel, ctl.MoveMode.CL_ABSOLUTE
        elif moveMode == "step": return d_handle, channel, ctl.MoveMode.STEP
        else: pass

    except ctl.Error as e:
        print("MCS2 {}: {}, error: {} (0x{:04X}) in line: {}. Press return to exit."
            .format(e.func, ctl.GetResultInfo(e.code), ctl.ErrorCode(e.code).name, e.code, (sys.exc_info()[-1].tb_lineno)))

    except Exception as ex:
        print("Unexpected error: {}, {} in line: {}".format(ex, type(ex), (sys.exc_info()[-1].tb_lineno)))
        raise


def moving_motor(d_handle, channel, move_mode, angleParameter):
    """
    Moves the motor to a specified position based on the movement mode and stores the position in a file.

    This function sets the initial position of the motor to zero and then moves the motor according to 
    the specified `move_mode` and `angleParameter`. After the movement is completed, it waits for the event 
    to confirm the movement and stores the final position in a file located in the specified path.

    Parameters:
    -----------
    d_handle : int
        The device handle representing the connected motor device.

    channel : int
        The channel of the device where the movement will be executed.

    move_mode : int
        The mode of movement, which determines how the motor moves:
        - `ctl.MoveMode.CL_ABSOLUTE`: Moves the motor to an absolute position.
        - `ctl.MoveMode.CL_RELATIVE`: Moves the motor relative to its current position.
        - `ctl.MoveMode.STEP`: Performs a step movement.

    angleParameter : int or float
        The target position or movement value, depending on the movement mode. It is converted to an integer 
        for compatibility with the movement function.

    path : str
        The path to the path where the motor’s position will be stored. The position is saved in a file 
        named `positions.txt` within this path.

    Returns:
    --------
    None

    Notes:
    ------
    - The function first sets the motor position to zero before initiating the movement.
    - The `move` function is called with the specified mode and angle parameter to execute the movement.
    - The function waits for the movement event to complete using `waitForEvent`.
    - The final position of the motor is retrieved and stored in a file using the `positions_storing` function.
    - Ensure that the specified path exists; otherwise, `positions_storing` will not create the file.

    Example:
    --------
    moving_motor(device_handle, 1, ctl.MoveMode.CL_ABSOLUTE, 500000, "/path/to/path")
    """
    ctl.SetProperty_i64(d_handle, channel, ctl.PropertyKey.POSITION, 0)
    move(d_handle, channel, move_mode, angleParameter)
    waitForEvent(d_handle)


def closing_motor(d_handle):
    """
    Closes the connection to the motor device if it is currently open.

    This function checks if the device handle (`d_handle`) is not `None`. If a valid handle 
    is detected, it disconnects the device and closes the connection using the SmarActCTL library. 
    A message is printed to indicate that the device has been disconnected.

    Parameters:
    -----------
    d_handle : int or None
        The device handle representing the connected motor device. If `None`, the function 
        does nothing.

    Returns:
    --------
    None

    Notes:
    ------
    - The function checks if `d_handle` is not `None` to determine if a device connection 
      is currently active. If so, it calls `ctl.Close` to safely close the connection.
    - A message, "device disconnected", is printed to confirm the disconnection.

    Example:
    --------
    closing_motor(device_handle)
    """
    if d_handle != None:
        ctl.Close(d_handle)
        print("Device closed")


