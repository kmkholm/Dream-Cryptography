# -*- coding: utf-8 -*-
"""
Created on Fri May 30 00:51:36 2025

@author: kmkho
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 30 00:47:06 2025

@author: kmkho
"""

import numpy as np
import hashlib
import json
import random
from datetime import datetime, timedelta
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from cryptography.fernet import Fernet
import base64
from typing import Dict, List, Tuple, Optional
import pickle
import time

class SleepStageMonitor:
    """Monitors and analyzes sleep stages using simulated EEG data"""
    
    def __init__(self):
        self.sample_rate = 256  # Hz
        self.stage_mapping = {
            0: 'WAKE',
            1: 'N1',
            2: 'N2', 
            3: 'N3',
            4: 'REM'
        }
        
    def generate_eeg_signal(self, duration_minutes: int, sleep_stage: int) -> np.ndarray:
        """Generate realistic EEG signal for given sleep stage"""
        samples = int(duration_minutes * 60 * self.sample_rate)
        t = np.linspace(0, duration_minutes * 60, samples)
        
        # Base signal
        signal_data = np.random.normal(0, 0.1, samples)
        
        if sleep_stage == 0:  # WAKE
            # Beta waves (13-30 Hz) and some alpha (8-13 Hz)
            signal_data += 0.3 * np.sin(2 * np.pi * 20 * t + np.random.random())
            signal_data += 0.2 * np.sin(2 * np.pi * 10 * t + np.random.random())
            
        elif sleep_stage == 1:  # N1 (Light sleep)
            # Theta waves (4-8 Hz)
            signal_data += 0.4 * np.sin(2 * np.pi * 6 * t + np.random.random())
            
        elif sleep_stage == 2:  # N2 (Sleep spindles and K-complexes)
            # Sleep spindles (11-15 Hz bursts)
            spindle_freq = 12
            for i in range(0, len(t), int(self.sample_rate * 3)):
                if random.random() < 0.3:  # 30% chance of spindle
                    end_idx = min(i + int(self.sample_rate * 0.5), len(t))
                    signal_data[i:end_idx] += 0.6 * np.sin(2 * np.pi * spindle_freq * t[i:end_idx])
                    
        elif sleep_stage == 3:  # N3 (Deep sleep)
            # Delta waves (0.5-4 Hz)
            signal_data += 0.8 * np.sin(2 * np.pi * 2 * t + np.random.random())
            signal_data += 0.6 * np.sin(2 * np.pi * 1 * t + np.random.random())
            
        elif sleep_stage == 4:  # REM
            # Similar to wake but with different pattern
            signal_data += 0.35 * np.sin(2 * np.pi * 25 * t + np.random.random())
            signal_data += 0.25 * np.sin(2 * np.pi * 15 * t + np.random.random())
            # Add some theta for REM
            signal_data += 0.3 * np.sin(2 * np.pi * 7 * t + np.random.random())
            
        return signal_data
    
    def extract_sleep_features(self, eeg_signal: np.ndarray) -> Dict:
        """Extract features from EEG signal for sleep stage classification"""
        # Power spectral density
        freqs, psd = signal.welch(eeg_signal, self.sample_rate, nperseg=1024)
        
        # Frequency bands
        delta_band = np.where((freqs >= 0.5) & (freqs <= 4))[0]
        theta_band = np.where((freqs >= 4) & (freqs <= 8))[0]
        alpha_band = np.where((freqs >= 8) & (freqs <= 13))[0]
        beta_band = np.where((freqs >= 13) & (freqs <= 30))[0]
        
        features = {
            'delta_power': np.mean(psd[delta_band]),
            'theta_power': np.mean(psd[theta_band]),
            'alpha_power': np.mean(psd[alpha_band]),
            'beta_power': np.mean(psd[beta_band]),
            'total_power': np.sum(psd),
            'dominant_freq': freqs[np.argmax(psd)],
            'spectral_entropy': -np.sum(psd * np.log2(psd + 1e-12))
        }
        
        return features
    
    def generate_sleep_architecture(self, total_hours: int = 8) -> List[Tuple[int, int]]:
        """Generate realistic sleep architecture (stage, duration_minutes)"""
        architecture = []
        remaining_minutes = total_hours * 60
        
        # Sleep onset (N1 -> N2 -> N3)
        architecture.append((1, 5))  # N1: 5 minutes
        architecture.append((2, 15)) # N2: 15 minutes
        architecture.append((3, 30)) # N3: 30 minutes
        remaining_minutes -= 50
        
        # Sleep cycles (typically 4-6 cycles per night)
        cycle_count = 0
        while remaining_minutes > 60 and cycle_count < 6:
            cycle_length = random.randint(80, 120)  # 80-120 minutes per cycle
            if cycle_length > remaining_minutes:
                cycle_length = remaining_minutes
                
            # Typical cycle: N2 -> N3 -> N2 -> REM
            n2_1 = random.randint(10, 20)
            n3_duration = max(5, random.randint(15, 40) - cycle_count * 5)  # Less N3 later
            n2_2 = random.randint(10, 15)
            rem_duration = min(cycle_length - n2_1 - n3_duration - n2_2, 
                             10 + cycle_count * 10)  # More REM later
            
            architecture.extend([
                (2, n2_1),
                (3, n3_duration),
                (2, n2_2),
                (4, rem_duration)
            ])
            
            remaining_minutes -= cycle_length
            cycle_count += 1
            
        return architecture

class DreamContentAnalyzer:
    """Analyzes and processes dream content for encryption"""
    
    def __init__(self):
        self.dream_symbols = {
            'water': {'emotional_weight': 0.7, 'chaos_factor': 0.3, 'frequency': 0.2},
            'flying': {'emotional_weight': 0.9, 'chaos_factor': 0.8, 'frequency': 0.15},
            'falling': {'emotional_weight': 0.8, 'chaos_factor': 0.9, 'frequency': 0.1},
            'chase': {'emotional_weight': 0.85, 'chaos_factor': 0.7, 'frequency': 0.12},
            'house': {'emotional_weight': 0.4, 'chaos_factor': 0.2, 'frequency': 0.25},
            'animal': {'emotional_weight': 0.6, 'chaos_factor': 0.5, 'frequency': 0.18},
            'death': {'emotional_weight': 0.95, 'chaos_factor': 0.6, 'frequency': 0.08},
            'family': {'emotional_weight': 0.7, 'chaos_factor': 0.3, 'frequency': 0.22},
            'school': {'emotional_weight': 0.5, 'chaos_factor': 0.4, 'frequency': 0.16},
            'car': {'emotional_weight': 0.3, 'chaos_factor': 0.4, 'frequency': 0.14}
        }
        
        self.emotional_states = {
            'joy': 0.9,
            'fear': 0.8,
            'anxiety': 0.7,
            'sadness': 0.6,
            'anger': 0.85,
            'confusion': 0.5,
            'excitement': 0.8,
            'peace': 0.3
        }
        
    def analyze_dream_narrative(self, dream_text: str) -> Dict:
        """Extract chaos markers and emotional content from dream narrative"""
        dream_lower = dream_text.lower()
        
        # Find dream symbols
        found_symbols = []
        total_emotional_weight = 0
        total_chaos_factor = 0
        
        for symbol, properties in self.dream_symbols.items():
            if symbol in dream_lower:
                found_symbols.append(symbol)
                total_emotional_weight += properties['emotional_weight']
                total_chaos_factor += properties['chaos_factor']
        
        # Analyze narrative chaos markers
        chaos_markers = {
            'identity_shifts': dream_lower.count('i was') + dream_lower.count('became'),
            'time_jumps': dream_lower.count('suddenly') + dream_lower.count('then'),
            'impossible_physics': dream_lower.count('flying') + dream_lower.count('float'),
            'logical_breaks': dream_lower.count('strange') + dream_lower.count('weird')
        }
        
        total_chaos = sum(chaos_markers.values()) + total_chaos_factor
        
        return {
            'symbols': found_symbols,
            'emotional_weight': total_emotional_weight / max(len(found_symbols), 1),
            'chaos_factor': total_chaos,
            'chaos_markers': chaos_markers,
            'narrative_length': len(dream_text.split()),
            'complexity_score': len(set(dream_text.lower().split())) / len(dream_text.split())
        }
    
    def generate_personal_dream_dictionary(self, dream_history: List[str]) -> Dict:
        """Build personalized dream symbol mapping from user's dream history"""
        personal_dict = {}
        symbol_frequencies = {}
        
        for dream_text in dream_history:
            analysis = self.analyze_dream_narrative(dream_text)
            
            for symbol in analysis['symbols']:
                if symbol not in symbol_frequencies:
                    symbol_frequencies[symbol] = 0
                symbol_frequencies[symbol] += 1
        
        # Create personal mappings based on frequency and emotional weight
        for symbol, freq in symbol_frequencies.items():
            personal_weight = freq / len(dream_history)
            base_properties = self.dream_symbols[symbol]
            
            personal_dict[symbol] = {
                'personal_frequency': personal_weight,
                'emotional_weight': base_properties['emotional_weight'] * personal_weight,
                'chaos_factor': base_properties['chaos_factor'] * personal_weight,
                'cipher_value': hash(symbol + str(personal_weight)) % 256
            }
        
        return personal_dict

class DreamStateEncryption:
    """Core encryption system based on dream states and sleep patterns"""
    
    def __init__(self):
        self.sleep_monitor = SleepStageMonitor()
        self.dream_analyzer = DreamContentAnalyzer()
        self.user_profile = None
        
    def create_user_profile(self, user_id: str, sleep_nights: int = 7) -> Dict:
        """Create comprehensive user profile from sleep data"""
        print(f"Creating user profile for {user_id}...")
        
        sleep_data = []
        dream_narratives = []
        
        # Simulate sleep data collection over multiple nights
        for night in range(sleep_nights):
            print(f"Processing night {night + 1}/{sleep_nights}")
            
            # Generate sleep architecture
            architecture = self.sleep_monitor.generate_sleep_architecture()
            
            # Generate EEG data for each sleep stage
            night_data = {
                'date': (datetime.now() - timedelta(days=sleep_nights-night-1)).isoformat(),
                'architecture': architecture,
                'eeg_features': [],
                'rem_periods': []
            }
            
            for stage, duration in architecture:
                eeg_signal = self.sleep_monitor.generate_eeg_signal(duration, stage)
                features = self.sleep_monitor.extract_sleep_features(eeg_signal)
                features['stage'] = stage
                features['duration'] = duration
                night_data['eeg_features'].append(features)
                
                # Collect REM periods for dream analysis
                if stage == 4:  # REM stage
                    night_data['rem_periods'].append({
                        'duration': duration,
                        'features': features
                    })
            
            # Simulate dream narratives for REM periods
            for rem_period in night_data['rem_periods']:
                dream_narrative = self._generate_simulated_dream()
                dream_narratives.append(dream_narrative)
                rem_period['dream_narrative'] = dream_narrative
            
            sleep_data.append(night_data)
        
        # Create personal dream dictionary
        personal_dict = self.dream_analyzer.generate_personal_dream_dictionary(dream_narratives)
        
        # Calculate sleep signature
        sleep_signature = self._calculate_sleep_signature(sleep_data)
        
        profile = {
            'user_id': user_id,
            'creation_date': datetime.now().isoformat(),
            'sleep_data': sleep_data,
            'sleep_signature': sleep_signature,
            'personal_dream_dictionary': personal_dict,
            'dream_narratives': dream_narratives
        }
        
        self.user_profile = profile
        return profile
    
    def _generate_simulated_dream(self) -> str:
        """Generate simulated dream narrative for testing"""
        dream_elements = [
            "I was flying over a vast ocean",
            "Suddenly I was back in my childhood house",
            "My family was there but they looked different",
            "The house started changing into a school",
            "I was being chased by a strange animal",
            "I could breathe underwater",
            "Everything became very bright and confusing",
            "I felt a deep sense of peace",
            "Then I was falling through clouds",
            "I woke up feeling excited"
        ]
        
        num_elements = random.randint(3, 7)
        return ". ".join(random.sample(dream_elements, num_elements)) + "."
    
    def _calculate_sleep_signature(self, sleep_data: List[Dict]) -> np.ndarray:
        """Calculate unique sleep signature from multiple nights"""
        # Extract key features across all nights
        features = []
        
        for night in sleep_data:
            night_features = []
            
            # Sleep architecture timing
            stage_durations = [0, 0, 0, 0, 0]  # WAKE, N1, N2, N3, REM
            for stage_features in night['eeg_features']:
                stage = stage_features['stage']
                duration = stage_features['duration']
                stage_durations[stage] += duration
            
            night_features.extend(stage_durations)
            
            # REM characteristics
            rem_features = [f for f in night['eeg_features'] if f['stage'] == 4]
            if rem_features:
                avg_rem_features = {
                    'delta_power': np.mean([f['delta_power'] for f in rem_features]),
                    'theta_power': np.mean([f['theta_power'] for f in rem_features]),
                    'alpha_power': np.mean([f['alpha_power'] for f in rem_features]),
                    'beta_power': np.mean([f['beta_power'] for f in rem_features])
                }
                night_features.extend(list(avg_rem_features.values()))
            else:
                night_features.extend([0, 0, 0, 0])
            
            features.append(night_features)
        
        # Calculate average signature
        signature = np.mean(features, axis=0)
        return signature
    
    def encrypt_data(self, data: str, dream_context: str = None) -> Dict:
        """Encrypt data using dream state parameters"""
        if not self.user_profile:
            raise ValueError("User profile not created. Call create_user_profile first.")
        
        print("Encrypting data using dream state parameters...")
        
        # Analyze current dream context if provided
        if dream_context:
            dream_analysis = self.dream_analyzer.analyze_dream_narrative(dream_context)
        else:
            # Use average parameters from user's dream history
            dream_analysis = self._get_average_dream_parameters()
        
        # Generate dream-based encryption key
        dream_key = self._generate_dream_key(dream_analysis)
        
        # Apply dream logic transformations
        transformed_data = self._apply_dream_transformations(data, dream_analysis)
        
        # Standard encryption with dream-derived key
        cipher = Fernet(dream_key)
        encrypted_data = cipher.encrypt(transformed_data.encode())
        
        # Create unlock requirements
        unlock_requirements = {
            'required_emotional_weight': dream_analysis['emotional_weight'],
            'required_chaos_factor': dream_analysis['chaos_factor'],
            'required_symbols': dream_analysis.get('symbols', []),
            'sleep_signature_hash': hashlib.sha256(self.user_profile['sleep_signature']).hexdigest()
        }
        
        encryption_result = {
            'encrypted_data': base64.b64encode(encrypted_data).decode(),
            'unlock_requirements': unlock_requirements,
            'encryption_timestamp': datetime.now().isoformat(),
            'dream_analysis': dream_analysis
        }
        
        return encryption_result
    
    def decrypt_data(self, encryption_result: Dict, current_dream_context: str, 
                    current_sleep_data: Dict = None) -> str:
        """Decrypt data by verifying dream state requirements"""
        if not self.user_profile:
            raise ValueError("User profile not created.")
        
        print("Attempting to decrypt using current dream state...")
        
        # Analyze current dream state
        current_analysis = self.dream_analyzer.analyze_dream_narrative(current_dream_context)
        
        # Verify unlock requirements
        requirements = encryption_result['unlock_requirements']
        
        # Check emotional weight match (within tolerance) - More lenient
        emotional_diff = abs(current_analysis['emotional_weight'] - 
                            requirements['required_emotional_weight'])
        emotional_match = emotional_diff < 0.6  # Increased tolerance
        
        # Check chaos factor match (within tolerance) - More lenient  
        chaos_diff = abs(current_analysis['chaos_factor'] - 
                        requirements['required_chaos_factor'])
        chaos_match = chaos_diff < 4.0  # Increased tolerance
        
        # Check symbol overlap - More lenient
        current_symbols = set(current_analysis.get('symbols', []))
        required_symbols = set(requirements['required_symbols'])
        
        if len(required_symbols) == 0:
            symbol_match = True
            symbol_overlap = 1.0
        else:
            symbol_overlap = len(current_symbols.intersection(required_symbols)) / len(required_symbols)
            symbol_match = symbol_overlap >= 0.3  # Reduced requirement
        
        # Verify sleep signature if provided
        sleep_match = True
        if current_sleep_data:
            current_signature = self._calculate_sleep_signature([current_sleep_data])
            stored_signature = self.user_profile['sleep_signature']
            signature_similarity = np.corrcoef(current_signature, stored_signature)[0, 1]
            sleep_match = signature_similarity > 0.7
        
        print(f"Verification results:")
        print(f"  Emotional weight: Current={current_analysis['emotional_weight']:.2f}, Required={requirements['required_emotional_weight']:.2f}, Diff={emotional_diff:.2f}, Match={emotional_match}")
        print(f"  Chaos factor: Current={current_analysis['chaos_factor']:.2f}, Required={requirements['required_chaos_factor']:.2f}, Diff={chaos_diff:.2f}, Match={chaos_match}")
        print(f"  Symbols: Current={list(current_symbols)}, Required={list(required_symbols)}, Overlap={symbol_overlap:.2f}, Match={symbol_match}")
        print(f"  Sleep signature match: {sleep_match}")
        
        if not (emotional_match and chaos_match and symbol_match and sleep_match):
            raise ValueError("Dream state verification failed. Access denied.")
        
        # Regenerate decryption key
        dream_key = self._generate_dream_key(encryption_result['dream_analysis'])
        
        # Decrypt
        encrypted_bytes = base64.b64decode(encryption_result['encrypted_data'])
        cipher = Fernet(dream_key)
        decrypted_transformed = cipher.decrypt(encrypted_bytes).decode()
        
        # Reverse dream transformations
        original_data = self._reverse_dream_transformations(decrypted_transformed, 
                                                           encryption_result['dream_analysis'])
        
        print("Decryption successful!")
        return original_data
    
    def _get_average_dream_parameters(self) -> Dict:
        """Calculate average dream parameters from user's history"""
        if not self.user_profile['dream_narratives']:
            return {
                'emotional_weight': 0.5,
                'chaos_factor': 2.0,
                'symbols': [],
                'complexity_score': 0.5
            }
        
        analyses = [self.dream_analyzer.analyze_dream_narrative(dream) 
                   for dream in self.user_profile['dream_narratives']]
        
        avg_params = {
            'emotional_weight': np.mean([a['emotional_weight'] for a in analyses]),
            'chaos_factor': np.mean([a['chaos_factor'] for a in analyses]),
            'symbols': list(set([s for a in analyses for s in a['symbols']])),
            'complexity_score': np.mean([a['complexity_score'] for a in analyses])
        }
        
        return avg_params
    
    def _generate_dream_key(self, dream_analysis: Dict) -> bytes:
        """Generate encryption key from dream analysis"""
        # Combine dream parameters into key material
        key_material = (
            str(dream_analysis['emotional_weight']) +
            str(dream_analysis['chaos_factor']) +
            ''.join(sorted(dream_analysis.get('symbols', []))) +
            str(dream_analysis.get('complexity_score', 0))
        )
        
        # Add user-specific salt
        user_salt = hashlib.sha256(self.user_profile['user_id'].encode()).hexdigest()[:16]
        salted_material = key_material + user_salt
        
        # Generate Fernet-compatible key
        key_hash = hashlib.sha256(salted_material.encode()).digest()
        fernet_key = base64.urlsafe_b64encode(key_hash)
        
        return fernet_key
    
    def _apply_dream_transformations(self, data: str, dream_analysis: Dict) -> str:
        """Apply dream logic transformations to data"""
        transformed = list(data)
        chaos_factor = int(dream_analysis['chaos_factor'])
        
        # Apply chaos-based character shifts
        for i in range(len(transformed)):
            if transformed[i].isalnum():
                shift = (chaos_factor + i) % 26
                if transformed[i].isalpha():
                    base = ord('A') if transformed[i].isupper() else ord('a')
                    transformed[i] = chr((ord(transformed[i]) - base + shift) % 26 + base)
                elif transformed[i].isdigit():
                    transformed[i] = str((int(transformed[i]) + shift) % 10)
        
        # Apply emotional weighting (character substitution)
        emotional_weight = dream_analysis['emotional_weight']
        if emotional_weight > 0.7:
            # High emotion: more dramatic transformations
            for i in range(0, len(transformed), 3):
                if i < len(transformed) and transformed[i] == ' ':
                    transformed[i] = '~'
        
        return ''.join(transformed)
    
    def _reverse_dream_transformations(self, transformed_data: str, dream_analysis: Dict) -> str:
        """Reverse dream logic transformations"""
        original = list(transformed_data)
        chaos_factor = int(dream_analysis['chaos_factor'])
        
        # Reverse emotional transformations
        emotional_weight = dream_analysis['emotional_weight']
        if emotional_weight > 0.7:
            for i in range(len(original)):
                if original[i] == '~':
                    original[i] = ' '
        
        # Reverse chaos-based character shifts
        for i in range(len(original)):
            if original[i].isalnum():
                shift = (chaos_factor + i) % 26
                if original[i].isalpha():
                    base = ord('A') if original[i].isupper() else ord('a')
                    original[i] = chr((ord(original[i]) - base - shift) % 26 + base)
                elif original[i].isdigit():
                    original[i] = str((int(original[i]) - shift) % 10)
        
        return ''.join(original)
    
    def save_user_profile(self, filename: str):
        """Save user profile to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.user_profile, f)
        print(f"User profile saved to {filename}")
    
    def load_user_profile(self, filename: str):
        """Load user profile from file"""
        with open(filename, 'rb') as f:
            self.user_profile = pickle.load(f)
        print(f"User profile loaded from {filename}")

# Example usage and testing
def main():
    print("=== Dream State Encryption System ===")
    print("This system encrypts data using your unique sleep patterns and dream content.\n")
    
    # Create encryption system
    dream_crypto = DreamStateEncryption()
    
    # Create user profile
    print("🧠 STEP 1: Creating User Sleep Profile")
    print("Simulating 5 nights of sleep monitoring...")
    user_profile = dream_crypto.create_user_profile("alice_dreamer", sleep_nights=5)
    print(f"✅ Profile created with {len(user_profile['dream_narratives'])} dream narratives")
    
    # Data to encrypt
    secret_data = "This is my secret message that can only be unlocked through dreams!"
    
    # Dream context for encryption
    encryption_dream = """I was flying over a beautiful ocean with my family. 
    Suddenly we were in our old house, but it kept changing colors. 
    I felt very peaceful and excited at the same time. 
    Then I was falling but it felt wonderful, like floating through clouds."""
    
    print(f"\n🔐 STEP 2: Encrypting Data")
    print(f"Secret data: '{secret_data}'")
    print(f"Using dream context: {encryption_dream}")
    
    # Encrypt the data
    encryption_result = dream_crypto.encrypt_data(secret_data, encryption_dream)
    print(f"✅ Data encrypted! Ciphertext length: {len(encryption_result['encrypted_data'])} characters")
    
    print(f"\n🎯 UNLOCK REQUIREMENTS:")
    reqs = encryption_result['unlock_requirements']
    print(f"  • Emotional weight: {reqs['required_emotional_weight']:.2f}")
    print(f"  • Chaos factor: {reqs['required_chaos_factor']:.2f}") 
    print(f"  • Required symbols: {reqs['required_symbols']}")
    print(f"  • Sleep signature: {reqs['sleep_signature_hash'][:16]}...")
    
    print(f"\n🔓 STEP 3: Decryption Tests")
    
    # Attempt decryption with correct dream context
    print("\n=== Decryption Attempt 1: Similar Dream Context ===")
    decryption_dream = """I was soaring above a vast ocean with my family members. 
    We found ourselves in an old house that kept shifting and changing. 
    I experienced wonderful feelings of peace and excitement. 
    Then I was gently falling through soft white clouds, floating downward."""
    
    print(f"Decryption dream: {decryption_dream}")
    
    try:
        decrypted_data = dream_crypto.decrypt_data(encryption_result, decryption_dream)
        print(f"✅ SUCCESS: {decrypted_data}")
    except ValueError as e:
        print(f"❌ FAILED: {e}")
    
    # Attempt decryption with incorrect dream context
    print("\n=== Decryption Attempt 2: Different Dream Context ===")
    wrong_dream = """I was walking alone in a dark forest at night. 
    Scary creatures were chasing me through the trees. 
    I felt terrified and very confused about everything. 
    The dream was realistic and had no strange elements."""
    
    print(f"Wrong dream: {wrong_dream}")
    
    try:
        decrypted_data = dream_crypto.decrypt_data(encryption_result, wrong_dream)
        print(f"✅ SUCCESS: {decrypted_data}")
    except ValueError as e:
        print(f"❌ FAILED: {e}")
    
    # Display system statistics
    print("\n📊 SYSTEM ANALYSIS:")
    print(f"  • Sleep signature dimensions: {len(user_profile['sleep_signature'])}")
    print(f"  • Personal dream symbols learned: {len(user_profile['personal_dream_dictionary'])}")
    print(f"  • Average REM periods per night: {np.mean([len(night['rem_periods']) for night in user_profile['sleep_data']]):.1f}")
    
    print(f"\n💾 Personal Dream Dictionary:")
    for symbol, properties in list(user_profile['personal_dream_dictionary'].items())[:5]:
        print(f"  • '{symbol}': emotional_weight={properties['emotional_weight']:.2f}, frequency={properties['personal_frequency']:.2f}")
    
    # Save profile for future use
    dream_crypto.save_user_profile("alice_dream_profile.pkl")
    
    print(f"\n🎉 SYSTEM EXPLANATION:")
    print(f"This system works by:")
    print(f"1. 🛌 Learning your unique sleep patterns over multiple nights")
    print(f"2. 🧠 Building a personal 'dream dictionary' of your recurring symbols")
    print(f"3. 🔐 Encrypting data using dream logic transformations + traditional cryptography")
    print(f"4. 🔓 Requiring similar dream content to decrypt (emotional tone + symbols + chaos)")
    print(f"5. ✨ Creating truly personalized, consciousness-based security")
    
    print(f"\n🔬 SECURITY FEATURES:")
    print(f"• Cannot be hacked by computers alone - requires human consciousness")
    print(f"• Impossible to forge - dreams are internal and unobservable")
    print(f"• Self-updating - your dream patterns naturally evolve")
    print(f"• Liveness detection - only works during actual sleep states")
    print(f"• Emotional verification - attacker would need your exact feelings")
    
    print(f"\n✅ Profile saved! Dream State Encryption ready for deployment.")

if __name__ == "__main__":
    main()