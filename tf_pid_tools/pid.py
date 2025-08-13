import numpy as np
import control as ct


class PID:

    _Kp : float = 0.0
    _Ki : float = 0.0
    _Kd : float = 0.0
    name : str | None = None
    tf: ct.TransferFunction | None
    integral : np.ndarray
    prev_error : np.ndarray

    def __init__(self, Kp: float=0.0, Ki: float=0.0, Kd: float=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = np.zeros((1,))
        self.prev_error = np.zeros((1,))

    #############################
    # Functions
    ##############################

    def update_params(self, params:list, update_name:bool=True) -> None:
        """
        Update PID parameters from parameters based on a cost function.
        Do not modify the nature of the PID controller, just update the parameters and transfer function.
        """
        if self.name == 'P':
            self._Kp = params[0]
        elif self.name == 'I':
            self._Ki = params[0]
        elif self.name == 'D':
            self._Kd = params[0]
        elif self.name == 'PI':
            self._Kp, self._Ki = params[0], params[1]
        elif self.name == 'PD':
            self._Kp, self._Kd = params[0], params[1]
        elif self.name == 'ID':
            self._Ki, self._Kd = params[0], params[1]
        elif self.name == 'PID':
            self._Kp, self._Ki, self._Kd = params[0], params[1], params[2]
        else:
            raise ValueError("Unknown PID controller type. Cannot update parameters.")
        if update_name:
            self._update_name()
        self._update_tf()

    def get_params(self) -> list:
        """
        Extracts PID parameters from parameters based on a cost function.
        """
        if self.name == 'P':
            return [self.Kp]
        elif self.name == 'I':
            return [self.Ki]
        elif self.name == 'D':
            return [self.Kd]
        elif self.name == 'PI':
            return [self.Kp, self.Ki]
        elif self.name == 'PD':
            return [self.Kp, self.Kd]
        elif self.name == 'ID':
            return [self.Ki, self.Kd]
        elif self.name == 'PID':
            return [self.Kp, self.Ki, self.Kd]
        else:
            raise ValueError("Unknown PID controller type. Cannot extract parameters.")

    def compute_command_batch(self, errors:np.ndarray, dt:float) -> np.ndarray:
        """
        Compute the control command history based on the error signal e and time step dt.
        This method implements the PID control algorithm.
        """
        integral = np.cumsum(errors) * dt
        derivative = np.gradient(errors, dt)

        u = self._Kp * errors + self._Ki * integral + self._Kd * derivative
        return u

    def compute_command(self, error:np.ndarray, dt:float) -> np.ndarray:
        """
        Compute the control command based on the error signal e and time step dt.
        This method implements the PID control algorithm.
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        u = self._Kp * error[-1] + self._Ki * self.integral + self._Kd * derivative
        return u

    def reset(self) -> None:
        """
        Resets the integral and derivative terms to zero.
        This is useful when starting a new control loop.
        """
        self.integral = np.zeros((1,))
        self.prev_error = np.zeros((1,))

    def print(self) -> None:
        """
        Prints the PID parameters and transfer function.
        """
        print(f"PID Controller: {self.name}")
        print(f"Kp: {self._Kp}, Ki: {self._Ki}, Kd: {self._Kd}")
        print(f"Transfer Function: {self.tf}")

    def copy(self):
        """
        Returns a deep copy of the PID controller with identical parameters and state.
        """
        new_pid = PID(self.Kp, self.Ki, self.Kd)
        new_pid.name = self.name
        new_pid.tf = self.tf  # La tf est immutable ici, donc référence OK
        new_pid.integral = self.integral.copy()
        new_pid.prev_error = self.prev_error.copy()
        return new_pid

    ###########################
    # Properties and Setters
    ###########################
    @property
    def Kp(self) -> float:
        """
        Returns the proportional gain Kp.
        """
        return self._Kp

    @Kp.setter
    def Kp(self, value:float) -> None:
        """
        Sets the proportional gain Kp.
        Must be non-negative.
        """
        if value < 0:
            raise ValueError("Kp must be non-negative")
        self._Kp = value
        self._update_name()
        self._update_tf()

    @property
    def Ki(self) -> float:
        """
        Returns the integral gain Ki.
        """
        return self._Ki

    @Ki.setter
    def Ki(self, value:float) -> None:
        """
        Sets the integral gain Ki.
        Must be non-negative.
        """
        if value < 0:
            raise ValueError("Ki must be non-negative")
        self._Ki = value
        self._update_name()
        self._update_tf()

    @property
    def Kd(self) -> float:
        """
        Returns the derivative gain Kd.
        """
        return self._Kd

    @Kd.setter
    def Kd(self, value:float) -> None:
        """
        Sets the derivative gain Kd.
        Must be non-negative.
        """
        if value < 0:
            raise ValueError("Kd must be non-negative")
        self._Kd = value
        self._update_name()
        self._update_tf()

    ############################
    # Updaters
    ############################

    def _update_tf(self) -> None:
        """
        Updates the transfer function based on the current PID parameters.
        """
        if self.name == 'P':
            self.tf = ct.tf([self._Kp], [1])
        elif self.name == 'I':
            self.tf = ct.tf([self._Ki], [1, 0])
        elif self.name == 'D':
            self.tf = ct.tf([self._Kd, 0], [1])
        elif self.name == 'PI':
            self.tf = ct.tf([self._Kp, self._Ki], [1, 0])
        elif self.name == 'PD':
            self.tf = ct.tf([self._Kd, self._Kp], [1])
        elif self.name == 'ID':
            self.tf = ct.tf([self._Kd, 0, self._Ki], [1, 0])
        elif self.name == 'PID':
            self.tf = ct.tf([self._Kd, self._Kp, self._Ki], [1, 0])
        else:
            self.tf = None

    def _update_name(self) -> None:
        """
        Updates the name of the PID controller based on the current parameters.
        """
        if self._Kp != 0 and self._Ki == 0 and self._Kd == 0:
            self.name = 'P'
        elif self._Kp == 0 and self._Ki != 0 and self._Kd == 0:
            self.name = 'I'
        elif self._Kp == 0 and self._Ki == 0 and self._Kd != 0:
            self.name = 'D'
        elif self._Kp != 0 and self._Ki != 0 and self._Kd == 0:
            self.name = 'PI'
        elif self._Kp != 0 and self._Ki == 0 and self._Kd != 0:
            self.name = 'PD'
        elif self._Kp == 0 and self._Ki != 0 and self._Kd != 0:
            self.name = 'ID'
        elif self._Kp != 0 and self._Ki != 0 and self._Kd != 0:
            self.name = 'PID'
        else:
            self.Kp = 1
