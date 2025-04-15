# Dyadic Interaction Dataset Generator: Comprehensive Project Description

## 1. Project Overview

The Dyadic Interaction Dataset Generator is an advanced pipeline system designed to automatically extract high-quality multi-modal dyadic interaction data from YouTube videos. The system identifies video segments containing exactly two people interacting, extracts comprehensive motion parameters (body, hand, face) for each person, and aligns this with separated audio streams.

This project aims to create a valuable dataset for research in:
- Human behavior analysis
- Multimodal AI training
- Human-computer interaction
- Animation and virtual reality character development
- Conversational dynamics and non-verbal communication studies

## 2. Key Requirements

### 2.1 Core Functionality Requirements

1. **Two-Person Interaction Detection**
   - Automatically identify video segments where exactly two people are interacting
   - Filter out segments with one person, more than two people, or scene transitions
   - Maintain consistent tracking of each individual throughout the segment
   - Establish minimum segment length (30 frames) for meaningful interaction analysis

2. **Comprehensive Motion Capture**
   - Extract full-body pose parameters for each person
   - Capture detailed hand gestures and movements
   - Extract facial expressions and movements
   - Integrate all parameters into a unified SMPLX model representation
   - Ensure temporal consistency and smoothness in motion tracking

3. **Audio Processing and Attribution**
   - Associate speech segments with the correct speaking person
   - Separate mixed audio into individual streams for each speaker
   - Maintain temporal alignment between visual and audio data
   - Handle overlapping speech scenarios

4. **Data Quality and Validation**
   - Implement confidence metrics for all extracted parameters
   - Filter out low-quality or unreliable detections
   - Ensure cross-modal consistency in the final dataset
   - Provide visualization tools for quality assessment

### 2.2 Technical Requirements

1. **Performance Requirements**
   - Process videos with resolution of at least 720p (720×1280)
   - Support standard frame rates (30 fps default)
   - Operate on CUDA-capable GPU environment for neural network inference
   - Handle variable video lengths efficiently

2. **Compatibility Requirements**
   - Work with common video formats (MP4, AVI, MOV)
   - Support different audio configurations
   - Operate within a conda environment with appropriate dependencies
   - Export data in standard formats (JSON, PKL)

3. **Scalability Requirements**
   - Process multiple videos in batch mode
   - Allow incremental processing and resuming from checkpoints
   - Manage storage efficiently for large datasets
   - Provide configurable parameters for different use cases

## 3. Proposed Technical Solution

### 3.1 System Architecture

The system follows a modular pipeline architecture consisting of five main components:

1. **Preprocessing Module**
   - Video decoding and frame extraction
   - Scene change detection to segment videos
   - Frame normalization and standardization
   - Initial filtering of unsuitable content

2. **Person Detection and Tracking Module**
   - Person detection using YOLOv8
   - Temporal tracking using ByteTrack
   - Identity maintenance across frames
   - Selection of segments with exactly two people

3. **Motion Estimation Module**
   - Body pose estimation using 4D-Humans (HMR2)
   - Hand pose estimation using HAMER
   - Face parameter extraction using SPECTRE
   - Integration into unified SMPLX 2020 parameters

4. **Audio Processing Module**
   - Speaker detection using TalkNet
   - Audio-visual correlation for speaker identification
   - Audio separation into individual streams
   - Temporal alignment with visual data

5. **Data Integration Module**
   - Cross-modal temporal alignment
   - Quality filtering and validation
   - Dataset formatting and metadata generation
   - Result visualization and validation

### 3.2 Pipeline Data Flow Diagram

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                 │     │                     │     │                     │
│  YouTube Videos ├────►│  Video Preprocessing├────►│ Person Detection &  │
│                 │     │                     │     │     Tracking        │
└─────────────────┘     └─────────────────────┘     └─────────┬───────────┘
                                                             │
                                                             │
                                                             ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│  Data Integration   │◄────┤  Audio Processing   │◄────┤  Motion Estimation  │
│                     │     │                     │     │                     │
└─────────┬───────────┘     └─────────────────────┘     └─────────────────────┘
          │
          │
          ▼
┌─────────────────────┐
│                     │
│  Dataset Export     │
│                     │
└─────────────────────┘
```

### 3.3 Detailed Data Flow Diagram

```
┌───────────────────────────────────────────────────────────────────────────┐
│ VIDEO PREPROCESSING                                                        │
│                                                                           │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────┐            │
│  │ Extract      │    │ Scene Change   │    │ Frame           │            │
│  │ Frames/Audio ├───►│ Detection      ├───►│ Normalization   │            │
│  └──────────────┘    └────────────────┘    └────────┬────────┘            │
│                                                    │                      │
└────────────────────────────────────────────────────┼──────────────────────┘
                                                     │
                                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ PERSON DETECTION & TRACKING                                                │
│                                                                           │
│  ┌──────────────┐    ┌────────────────┐    ┌────────────────┐             │
│  │ YOLOv8       │    │ ByteTrack      │    │ Two-Person     │             │
│  │ Detection    ├───►│ Tracking       ├───►│ Segmentation   │             │
│  └──────────────┘    └────────────────┘    └────────┬───────┘             │
│                                                    │                      │
└────────────────────────────────────────────────────┼──────────────────────┘
                        ┌────────────────────────────┼─────────────────────┐
                        │                            │                     │
                        ▼                            ▼                     ▼
┌───────────────────────────────┐  ┌─────────────────────────┐  ┌──────────────────────┐
│ BODY POSE ESTIMATION          │  │ HAND POSE ESTIMATION    │  │ FACE ESTIMATION      │
│                               │  │                         │  │                      │
│  ┌──────────┐   ┌───────────┐ │  │  ┌──────────┐  ┌──────┐ │  │  ┌──────────┐  ┌───┐ │
│  │ Crop     │   │ 4D-Humans │ │  │  │ Hand     │  │HAMER │ │  │  │ Face     │  │SPE│ │
│  │ People   ├──►│ (HMR2)    │ │  │  │ Regions  ├─►│Model │ │  │  │ Regions  ├─►│CTR│ │
│  └──────────┘   └─────┬─────┘ │  │  └──────────┘  └──┬───┘ │  │  └──────────┘  └─┬─┘ │
│                       │       │  │                   │     │  │                  │   │
└───────────────────────┼───────┘  └───────────────────┼─────┘  └──────────────────┼───┘
                        │                              │                           │
                        ▼                              ▼                           ▼
                ┌─────────────────────────────────────────────────────────────────┐
                │               SMPLX MODEL INTEGRATION                           │
                │  ┌─────────────────────────────────────────────────────────┐    │
                │  │                                                         │    │
                │  │  Body + Hands + Face → Unified SMPLX Representation     │    │
                │  │                                                         │    │
                │  └───────────────────────────────┬─────────────────────────┘    │
                │                                  │                              │
                └──────────────────────────────────┼──────────────────────────────┘
                                                   │
                                                   │
      ┌───────────────────────────────────────────┐│
      │                                           ││
      ▼                                           ▼▼
┌───────────────────────────┐           ┌──────────────────────────┐
│ AUDIO PROCESSING          │           │ DATA INTEGRATION         │
│                           │           │                          │
│  ┌──────────┐ ┌─────────┐ │           │  ┌───────────────────┐   │
│  │ TalkNet  │ │ Audio   │ │           │  │ Temporal Alignment │   │
│  │ Speaker  ├►│ Source  ├─┼──────────►│  │ & Quality Filtering├──►│ DATASET EXPORT
│  │ Detection│ │ Separ.  │ │           │  └───────────────────┘   │
│  └──────────┘ └─────────┘ │           │                          │
│                           │           │                          │
└───────────────────────────┘           └──────────────────────────┘
```

### 3.4 Key Technologies

1. **Person Detection and Tracking**
   - YOLOv8 for efficient and accurate person detection
   - ByteTrack for robust multi-object tracking
   - Custom spatial-temporal consistency constraints

2. **Motion Capture Stack**
   - 4D-Humans/HMR2 for body pose estimation
   - HAMER for detailed hand pose estimation
   - SPECTRE for facial parameter extraction
   - SMPLX 2020 model for unified representation

3. **Audio Processing**
   - TalkNet for audio-visual speaker detection
   - Conv-TasNet for audio source separation
   - Custom alignment algorithms for AV synchronization

4. **Data Management**
   - Efficient caching and storage mechanisms
   - Parallel processing where possible
   - Checkpointing for long-running processes

## 4. Detailed Pipeline Stages

### 4.1 Video Preprocessing

**Input**: Raw video files (.mp4, .avi, etc.)
**Output**: Extracted frames, audio track, scene change markers
**Process**:
1. Decode video and extract frames at specified FPS (default: 30)
2. Normalize frame resolution to 720p (720×1280)
3. Detect scene changes using threshold-based comparison (threshold: 35.0)
4. Extract audio track for separate processing
5. Cache frames to disk for efficient reuse

**Data at this stage**:
- Frames: Sequence of normalized images (720×1280 RGB)
- Audio: Extracted audio track (16kHz WAV)
- Metadata: Video properties, scene change timestamps

### 4.2 Person Detection and Tracking

**Input**: Preprocessed video frames
**Output**: Person tracks with bounding boxes, two-person segments
**Process**:
1. Apply YOLOv8 detector to each frame to identify persons
2. Filter detections using confidence threshold (0.5) and size constraints:
   - Minimum bounding box area: 10,000 pixels
   - Minimum ratio of bbox to frame: 1%
   - Minimum person height: 20% of frame
   - Maximum ratio of bbox to frame: 40%
   - Maximum person height: 90% of frame
   - Maximum edge proximity: 5% of frame
   - Aspect ratio range: 0.3-0.9 (width/height)
3. Track persons across frames using ByteTrack
   - Max tracking age: 30 frames
   - Minimum hits to confirm: 3 frames
   - IOU threshold: 0.3
4. Identify segments with exactly two people present
5. Ensure segments meet minimum length requirement (30 frames)

**Data at this stage**:
- Person tracks: Dictionary mapping track IDs to series of bounding boxes
- Two-person segments: List of video segments with frame ranges
- Tracking metadata: Confidence scores, track statistics

### 4.3 Motion Estimation

**Input**: Person tracks, video frames
**Output**: Body, hand, face parameters for each person
**Process**:

#### 4.3.1 Body Pose Estimation
1. For each person track:
   - Crop person from frames using bounding boxes (with padding)
   - Process crops through 4D-Humans/HMR2 model in batches
   - Extract SMPL parameters (pose, shape, global orientation)
   - Apply temporal smoothing if enabled (window size: 5)

#### 4.3.2 Hand Pose Estimation
1. For each person track:
   - Detect hand regions using body pose information
   - Process hand crops through HAMER model
   - Extract hand pose parameters
   - Align with body pose coordinates

#### 4.3.3 Face Parameter Extraction
1. For each person track:
   - Detect face region using body pose information
   - Process face crops through SPECTRE model
   - Extract facial expression parameters
   - Align with body pose coordinates

#### 4.3.4 SMPLX Integration
1. For each person:
   - Combine body, hand, and face parameters
   - Convert to unified SMPLX 2020 representation
   - Ensure anatomical consistency
   - Compute confidence scores for different body parts

**Data at this stage**:
- Body parameters: SMPL pose, shape, global orientation
- Hand parameters: Hand articulation for both hands
- Face parameters: Facial expression coefficients
- Integrated SMPLX parameters: Full-body representation
- Confidence scores: Per-parameter reliability metrics

### 4.4 Audio Processing

**Input**: Audio track, person tracks, video frames
**Output**: Speaker identification, separated audio streams
**Process**:

#### 4.4.1 Speaker Detection
1. Process video through TalkNet model
   - Analyze audio-visual correlation
   - Identify speaking probability for each detected face
   - Apply confidence threshold (0.5)

#### 4.4.2 Speaker Association
1. For each person track:
   - Map TalkNet detections to tracked persons
   - Resolve ambiguities using spatial correspondence
   - Generate speaker activity timeline

#### 4.4.3 Audio Separation
1. For segments with identified speakers:
   - Apply Conv-TasNet separation model
   - Guide separation using speaker activity information
   - Generate individual audio streams

**Data at this stage**:
- Speaker activities: Timeline of speaking activity for each person
- Separated audio: Individual audio streams for each speaker
- Audio metadata: Confidence scores, quality metrics

### 4.5 Data Integration and Export

**Input**: All processed data from previous stages
**Output**: Structured dataset entries with unified representations
**Process**:
1. Align all modalities temporally
   - Use alignment window (3 frames) to handle slight desynchronization
   - Ensure consistent frame rate across modalities
2. Apply quality filtering
   - Use confidence threshold (0.3) to filter low-quality data
   - Ensure cross-modal consistency
3. Generate metadata and organize results
4. Export in specified format (JSON default)
   - Optionally compress results if enabled

**Final Data Structure**:
```
/simurgh/group/juze/processed_data/dyadic_interaction/results/
  └── video_name/
      ├── metadata.yaml  # Video metadata
      ├── segments/
      │   └── segment_XXX/  # One directory per two-person segment
      │       ├── segment_info.yaml  # Segment metadata
      │       ├── preview/  # Visualization frames
      │       ├── motion/  # Body, hand, face parameters
      │       │   ├── person_1_smplx.pkl
      │       │   └── person_2_smplx.pkl
      │       └── audio/  # Separated audio streams
      │           ├── person_1.wav
      │           └── person_2.wav
```

## 5. Implementation Approach

### 5.1 Development Phases

#### Phase 1: MVP Implementation
- Basic pipeline architecture with module interfaces
- Video preprocessing and two-person segment extraction
- Simple person tracking with bounding boxes
- Basic body pose estimation (4D-Humans/HMR2 only)
- Initial audio-person association using TalkNet
- Data export in simple JSON format
- Basic visualization for quality checking

#### Phase 2: Core Enhancement
- Integration of hand pose estimation (HAMER)
- Improved person tracking with re-identification
- Enhanced audio-visual association
- Basic audio separation
- Parallel processing for improved throughput
- Basic dataset statistics and quality metrics

#### Phase 3: Complete Feature Set
- Face parameter extraction and integration (SPECTRE)
- Full SMPLX model integration
- Advanced audio separation techniques
- Temporal consistency optimization
- Comprehensive data validation tools
- Extended dataset format options

#### Phase 4: Refinement and Tools
- Interactive visualization and data exploration tools
- Automated quality filtering
- Cross-modal consistency checks
- Dataset search and query capabilities
- Model fine-tuning tools for specific use cases

### 5.2 Dependency Management

The project uses a conda environment for dependency management:
- Python 3.10 as the base environment
- PyTorch for deep learning models
- OpenCV for image processing
- Additional libraries for audio processing, 3D modeling, and data handling

Dependencies are specified in the `environment.yml` file and installed via the provided installation script.

### 5.3 Configuration Management

The system uses a YAML-based configuration approach:
- Default settings in `configs/default.yaml`
- Command-line overrides for specific runs
- Modular configuration for each pipeline component
- Easy customization for different processing requirements

## 6. Technical Challenges and Mitigations

### 6.1 Tracking Robustness
**Challenge**: Maintaining consistent person tracking across scene changes, occlusions
**Mitigation**: 
- Conservative segmentation that prioritizes reliable tracking
- Spatial-temporal consistency constraints
- Multiple tracking fallback strategies

### 6.2 Model Integration
**Challenge**: Integrating different pose estimation models (body, hands, face)
**Mitigation**:
- Clear intermediate representations
- Validation steps between components
- Anatomical consistency checks in final integration

### 6.3 Audio Separation Quality
**Challenge**: Poor audio separation in noisy environments or overlapping speech
**Mitigation**:
- Quality metrics to filter out sequences with unreliable audio
- Guided separation using visual cues
- Confidence scores for separated audio streams

### 6.4 Computational Resources
**Challenge**: High computational demands for neural network inference
**Mitigation**:
- Batch processing and efficient resource management
- Checkpointing for long-running processes
- Configurable quality-performance tradeoffs

## 7. Expected Outcomes

### 7.1 Dataset Characteristics
- High-quality dyadic interaction segments from YouTube videos
- Comprehensive motion capture data in SMPLX format
- Separated audio streams aligned with visual data
- Rich metadata for research and analysis
- Suitable for training multimodal AI models

### 7.2 Performance Metrics
- Processing throughput: ~20-30 minutes processing per minute of video
- Pose estimation accuracy comparable to state-of-the-art methods
- Audio separation quality suitable for speech analysis
- Segment identification precision >90% for clear two-person interactions

### 7.3 Validation Approach
- Visual inspection of motion overlays on original video
- Audio playback of separated streams
- Cross-modal consistency checks
- Statistical analysis of dataset characteristics
- Sample applications using the generated data

## 8. Conclusion

The Dyadic Interaction Dataset Generator provides a comprehensive solution for automatically extracting high-quality multimodal interaction data from YouTube videos. By integrating state-of-the-art techniques in computer vision, audio processing, and 3D modeling, the system enables the creation of valuable datasets for research in human behavior analysis, multimodal AI, and human-computer interaction.

The modular architecture allows for incremental development and continuous improvement, while the configurable pipeline supports various research and application needs. With a focus on data quality and comprehensive representation, the generated datasets will provide a solid foundation for advancing our understanding of human dyadic interactions. 