import numpy as np
from scipy import signal

class WaveformGenerator:
    def __init__(self, amplitude=1.0, frequency=1.0, phase_shift=0.0, 
                 offset=0.0, signal_type='sin', noise_amplitude=0.0, noise_seed=None):
        """
        Initialize waveform generator with specified parameters
        
        Parameters:
        -----------
        amplitude : float
            Peak amplitude of the waveform
        frequency : float
            Frequency in Hz
        phase_shift : float
            Initial phase shift in radians
        offset : float
            DC offset of the waveform
        signal_type : str
            Type of waveform ('sin', 'saw', 'tri', 'square', 'ramp')
        noise_amplitude : float
            Peak amplitude of random noise
        noise_seed : int or None
            Seed for random number generator
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.offset = offset
        self.signal_type = signal_type.lower()
        self.noise_amplitude = noise_amplitude
        self.noise_seed = noise_seed if noise_seed is not None else np.random.randint(0, 1000000)
        
        # Initialize list to store composite signals
        self.composite_signals = []
        
        # Validate signal type
        valid_types = ['sin', 'saw', 'tri', 'square', 'ramp']
        if self.signal_type not in valid_types:
            raise ValueError(f"Signal type must be one of {valid_types}")
        
        # Initialize random number generator
        self.rng = np.random.RandomState(self.noise_seed)

    def add_signal(self, signal_generator):
        """
        Add another WaveformGenerator instance to be combined with this signal
        
        Parameters:
        -----------
        signal_generator : WaveformGenerator
            Another instance of WaveformGenerator to be combined
        """
        if not isinstance(signal_generator, WaveformGenerator):
            raise ValueError("Added signal must be an instance of WaveformGenerator")
        self.composite_signals.append(signal_generator)

    def remove_signal(self, index):
        """
        Remove a composite signal by index
        
        Parameters:
        -----------
        index : int
            Index of the signal to remove
        """
        if 0 <= index < len(self.composite_signals):
            self.composite_signals.pop(index)
        else:
            raise IndexError("Signal index out of range")

    def clear_composite_signals(self):
        """Remove all composite signals"""
        self.composite_signals = []

    def _generate_base_waveform(self, t):
        """Generate the base waveform without noise or quantization"""
        # Angular frequency
        w = 2 * np.pi * self.frequency
        
        # Phase adjusted time array
        t_phase = w * t + self.phase_shift
    
        if self.signal_type == 'sin':
            wave = np.sin(t_phase)
        elif self.signal_type == 'square':
            wave = signal.square(t_phase)
        elif self.signal_type == 'tri':
            wave = signal.sawtooth(t_phase, width=0.5)
        elif self.signal_type == 'saw':
            # Adjust phase to align peaks with sine
            wave = signal.sawtooth(t_phase + np.pi/2)
        elif self.signal_type == 'ramp':
            # Ramp is a sawtooth with phase adjustment to align peaks
            wave = signal.sawtooth(t_phase + np.pi/2, width=1.0)
            
        return wave

    def _add_noise(self, signal):
        """Add random noise to the signal"""
        if self.noise_amplitude > 0:
            noise = self.rng.normal(0, self.noise_amplitude/np.sqrt(2), size=signal.shape)
            return signal + noise
        return signal

    def _quantize(self, signal, bit_depth):
        """Quantize the signal to specified bit depth"""
        if bit_depth is None:
            return signal
            
        levels = 2**bit_depth
        max_val = np.max(np.abs(signal))
        
        # Scale to full range of bits
        scaled = signal * ((levels-1) / (2 * max_val))
        quantized = np.round(scaled)
        
        # Scale back to original range
        return quantized * (2 * max_val / (levels-1))

    def _combine_signals(self, t, base_signal, sample_rate, phase_shift_type, phase_shift_value):
        """Combine base signal with all composite signals"""
        combined_signal = base_signal.copy()
        
        # Add each composite signal
        for sig_gen in self.composite_signals:
            _, additional_signal = sig_gen.get_waveform(
                sample_rate=sample_rate,
                duration=t[-1] - t[0],
                phase_shift_type=phase_shift_type,
                phase_shift_value=phase_shift_value,
                combine_composite=False  # Prevent infinite recursion
            )
            combined_signal += additional_signal
            
        return combined_signal

    def get_waveform(self, sample_rate, duration=0, bit_depth=None, 
                     phase_shift_type='radians', phase_shift_value=None,
                     combine_composite=True, duration_type=None, force=False):
        """
        Generate waveform samples
        
        Parameters:
        -----------
        sample_rate : float
            Sampling rate in Hz
        duration : float
            Duration of the waveform in seconds
        bit_depth : int or None
            Number of bits for quantization (None for no quantization)
        phase_shift_type : str
            Type of phase shift ('radians', 'time', 'samples')
        phase_shift_value : float or None
            Additional phase shift value in specified units
        combine_composite : bool
            Whether to include composite signals in the output
        duration_type : str
            Selects  method of durations calculation. 
            Can be 'time' (default), 'samples', or 'cycles'.
        force: bool
            Overrides length limitations
        Returns:
        --------
        tuple : (time_array, signal_array)
            Arrays containing time points and corresponding signal values
        """
        # Sample soft limit
        soft_limit = int(1e5)

        # Generate time array
        if duration_type in [None,'time']:
            num_samples = int(sample_rate * duration)
        elif duration_type == 'samples':
            num_samples = duration
            duration = num_samples / sample_rate
            num_samples = int(num_samples)
        elif duration_type == 'cycles':
            duration = duration * 1/self.frequency
            num_samples = int(sample_rate * duration)
        if num_samples > soft_limit:
            raise ValueError(f"The requested waveform would generate {num_samples} samples (soft cap of {soft_limit}). Pass 'force=True' to get the data anyways.")
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Apply additional phase shift if specified
        total_phase = self.phase_shift
        if phase_shift_value is not None:
            if phase_shift_type == 'radians':
                total_phase += phase_shift_value
            elif phase_shift_type == 'time':
                total_phase += 2 * np.pi * self.frequency * phase_shift_value
            elif phase_shift_type == 'samples':
                total_phase += 2 * np.pi * self.frequency * (phase_shift_value / sample_rate)
            else:
                raise ValueError("phase_shift_type must be 'radians', 'time', or 'samples'")
        
        # Store original phase and temporarily set to total phase
        orig_phase = self.phase_shift
        self.phase_shift = total_phase
        
        # Generate base waveform
        signal = self._generate_base_waveform(t)
        
        # Restore original phase
        self.phase_shift = orig_phase
        
        # Apply amplitude and offset
        signal = self.amplitude * signal + self.offset
        
        # Add noise
        signal = self._add_noise(signal)
        
        # Combine with composite signals if requested
        if combine_composite and self.composite_signals:
            signal = self._combine_signals(t, signal, sample_rate, 
                                        phase_shift_type, phase_shift_value)
        
        # Apply quantization
        signal = self._quantize(signal, bit_depth)
        
        return t, signal
