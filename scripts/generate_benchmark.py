"""
Phase 28 — Research Mind 170-Question Evaluation Benchmark Generator

Creates evaluation/benchmark_dataset.json containing exactly 170 research queries:
- 30 Artificial Intelligence
- 30 Cybersecurity
- 30 Agriculture
- 30 Healthcare
- 30 Climate
- 10 Multi-domain (AI+HC, AG+CL, etc.)
- 10 Out-of-corpus / Online Fallback queries

Each question includes:
1. question: str
2. expected_domain: str or List[str]
3. expected_intent: str
4. relevant_paper_ids: List[str]
5. expected_evidence_criteria: str
"""

import json
from pathlib import Path

BASE_DIR = Path(r"c:\Users\nugur\Desktop\researchmind")
BENCHMARK_PATH = BASE_DIR / "evaluation" / "benchmark_dataset.json"

AI_QUESTIONS = [
    # Definition
    ("What is retrieval-augmented generation in large language models?", "artificial_intelligence", "Definition", ["AI001", "AI002"], "Must define RAG and mention external vector/document retrieval."),
    ("What is explainable AI and why is it important?", "artificial_intelligence", "Definition", ["AI005", "AI006"], "Must explain XAI concept and feature attribution."),
    ("What is a Transformer neural network architecture?", "artificial_intelligence", "Definition", ["AI010"], "Must define self-attention and transformer blocks."),
    
    # Mechanism
    ("How does self-attention compute token dependencies in transformers?", "artificial_intelligence", "Mechanism", ["AI011"], "Explain query, key, value matrix multiplications and softmax attention weights."),
    ("How do diffusion models generate high-fidelity images step-by-step?", "artificial_intelligence", "Mechanism", ["AI015"], "Explain forward noise addition and reverse denoising process."),
    ("How does policy gradient optimization work in deep reinforcement learning?", "artificial_intelligence", "Mechanism", ["AI020"], "Explain policy optimization, rewards, and action distribution updates."),
    
    # Advantages
    ("What are the primary advantages of parameter-efficient fine-tuning like LoRA?", "artificial_intelligence", "Advantages", ["AI025"], "Highlight reduced memory footprint, low-rank matrices, and fast adaptation."),
    ("What are the advantages of agentic AI workflows over single-prompt LLM generation?", "artificial_intelligence", "Advantages", ["AI030"], "Discuss multi-step reasoning, tool use, self-reflection, and execution loops."),
    ("What are the benefits of dense vector retrieval over keyword search in RAG systems?", "artificial_intelligence", "Advantages", ["AI035"], "Discuss semantic matching, synonym handling, and dense vector embeddings."),
    
    # Limitations / Challenges
    ("What are the major limitations of current large language models in logical reasoning?", "artificial_intelligence", "Limitations", ["AI040"], "Discuss hallucinations, context window limits, and error propagation."),
    ("What challenges impede real-time deployment of vision transformer models?", "artificial_intelligence", "Challenges", ["AI045"], "Discuss quadratic computational complexity, high memory overhead, and latency."),
    ("What are the main safety challenges in autonomous AI agent execution?", "artificial_intelligence", "Challenges", ["AI050"], "Discuss reward hacking, alignment drift, and unconstrained action execution."),
    
    # Applications
    ("How is computer vision applied in autonomous vehicle navigation?", "artificial_intelligence", "Applications", ["AI055"], "Discuss object detection, lane segmentation, and depth estimation."),
    ("How are graph neural networks applied to molecular structure prediction?", "artificial_intelligence", "Applications", ["AI060"], "Discuss node/edge representations for atoms and chemical bonds."),
    ("How is reinforcement learning used in robotic manipulation tasks?", "artificial_intelligence", "Applications", ["AI065"], "Discuss trajectory optimization, motor control, and feedback loops."),
    
    # Comparison
    ("Compare convolutional neural networks and vision transformers for image classification.", "artificial_intelligence", "Comparison", ["AI070"], "Compare inductive bias, patch embeddings, and computational efficiency."),
    ("Compare dense retrieval and hybrid BM25 vector retrieval in RAG architectures.", "artificial_intelligence", "Comparison", ["AI075"], "Compare exact keyword precision vs semantic recall."),
    ("Compare supervised fine-tuning and reinforcement learning from human feedback.", "artificial_intelligence", "Comparison", ["AI080"], "Compare token likelihood optimization vs human preference alignment."),
    
    # Evaluation / Detection / Prediction
    ("How are hallucinations evaluated in retrieval-augmented language models?", "artificial_intelligence", "Evaluation", ["AI085"], "Discuss claim verification, groundedness metrics, and NLI models."),
    ("How do explainability methods detect bias in deep neural networks?", "artificial_intelligence", "Detection", ["AI090"], "Discuss SHAP, LIME, and feature attribution heatmaps."),
    ("How can machine learning predict model drift in production AI pipelines?", "artificial_intelligence", "Prediction", ["AI095"], "Discuss feature distribution shifts and monitoring statistical metrics."),
    
    # Additional AI intent coverage up to 30
    ("What is deep reinforcement learning?", "artificial_intelligence", "Definition", ["AI100"], "Define Q-learning and deep neural policy networks."),
    ("How do generative adversarial networks train generator and discriminator models?", "artificial_intelligence", "Mechanism", ["AI105"], "Explain min-max adversarial loss game between generator and discriminator."),
    ("What are the limitations of zero-shot prompting in complex math problems?", "artificial_intelligence", "Limitations", ["AI110"], "Discuss lack of step-by-step scratchpads and arithmetic errors."),
    ("How is natural language processing applied to automated code synthesis?", "artificial_intelligence", "Applications", ["AI115"], "Discuss code LLMs, syntax trees, and unit test generation."),
    ("Evaluate the effectiveness of chain-of-thought prompting in LLM reasoning.", "artificial_intelligence", "Evaluation", ["AI120"], "Discuss step-by-step intermediate reasoning steps and accuracy gains."),
    ("How does contrastive learning pre-train visual representations?", "artificial_intelligence", "Mechanism", ["AI125"], "Explain positive and negative pair similarity losses like SimCLR/CLIP."),
    ("What are the advantages of sparse mixture-of-experts architectures?", "artificial_intelligence", "Advantages", ["AI130"], "Discuss active parameter routing, reduced compute per token, and scaling."),
    ("What challenges affect multi-agent coordination in game theory scenarios?", "artificial_intelligence", "Challenges", ["AI135"], "Discuss non-stationary environments and communication bottlenecks."),
    ("How is RAG evaluated for factual faithfulness?", "artificial_intelligence", "Evaluation", ["AI140"], "Discuss RAGAS, faithfulness scoring, and context relevance metrics.")
]

CY_QUESTIONS = [
    # Definition
    ("What is network intrusion detection and how does it safeguard enterprise systems?", "cybersecurity", "Definition", ["CY001", "CY002"], "Define IDS and distinguish signature vs anomaly based detection."),
    ("What is zero trust network architecture?", "cybersecurity", "Definition", ["CY005"], "Explain continuous authentication and least privilege access."),
    ("What is homomorphic encryption in privacy-preserving computation?", "cybersecurity", "Definition", ["CY010"], "Explain performing computations directly on encrypted ciphertext."),
    
    # Mechanism
    ("How does machine learning detect zero-day malware variants?", "cybersecurity", "Mechanism", ["CY015"], "Discuss dynamic API call sequence analysis and behavioral embeddings."),
    ("How do ransomware detection systems detect unauthorized file encryption in real time?", "cybersecurity", "Mechanism", ["CY020"], "Explain monitoring I/O entropy, file header changes, and shadow copy deletion."),
    ("How does multi-factor authentication prevent credential stuffing attacks?", "cybersecurity", "Mechanism", ["CY025"], "Explain combining password, TOTP, and hardware token verification."),
    
    # Advantages
    ("What are the key advantages of differential privacy in sensitive dataset analytics?", "cybersecurity", "Advantages", ["CY030"], "Discuss mathematical privacy guarantees and bounded noise injection."),
    ("What are the benefits of AI-driven automated threat response in SOC environments?", "cybersecurity", "Advantages", ["CY035"], "Discuss sub-second response times, playbook automation, and reduced triage fatigue."),
    ("What are the advantages of container isolation in cloud security?", "cybersecurity", "Advantages", ["CY040"], "Discuss process namespace separation and lightweight security boundaries."),
    
    # Limitations / Challenges
    ("What are the major limitations of signature-based intrusion detection systems?", "cybersecurity", "Limitations", ["CY045"], "Discuss inability to detect novel obfuscated attacks and zero-day exploits."),
    ("What challenges hinder widespread adoption of fully homomorphic encryption?", "cybersecurity", "Challenges", ["CY050"], "Discuss extreme computational overhead, noise expansion, and latency."),
    ("What are the primary security challenges in securing IoT edge devices?", "cybersecurity", "Challenges", ["CY055"], "Discuss constrained hardware, unpatched firmware, and weak default credentials."),
    
    # Applications
    ("How is deep learning applied to phishing URL and malicious email detection?", "cybersecurity", "Applications", ["CY060"], "Discuss character-level CNNs, domain age analysis, and NLP email parsing."),
    ("How is machine learning used in cyber threat intelligence forecasting?", "cybersecurity", "Applications", ["CY065"], "Discuss dark web forum scraping, IOC extraction, and threat actor profiling."),
    ("How is cryptography applied to secure hardware enclaves like Intel SGX?", "cybersecurity", "Applications", ["CY070"], "Discuss memory encryption, remote attestation, and isolated enclaves."),
    
    # Comparison
    ("Compare static code analysis and dynamic sandbox analysis for malware detection.", "cybersecurity", "Comparison", ["CY075"], "Compare code inspection without execution vs behavioral observation in sandbox."),
    ("Compare symmetric encryption and asymmetric cryptography in network protocols.", "cybersecurity", "Comparison", ["CY080"], "Compare key exchange overhead and computational speed."),
    ("Compare signature-based detection and anomaly-based detection in network firewalls.", "cybersecurity", "Comparison", ["CY085"], "Compare false positive rates and zero-day detection capability."),
    
    # Detection / Prediction / Evaluation
    ("How do neural networks detect adversarial perturbations in malware samples?", "cybersecurity", "Detection", ["CY090"], "Discuss adversarial training, gradient masking, and robust classifiers."),
    ("How can machine learning predict network vulnerability exploitation likelihood?", "cybersecurity", "Prediction", ["CY095"], "Discuss CVSS scoring, exploit database analysis, and threat telemetry."),
    ("Evaluate the effectiveness of user behavior analytics in detecting insider threats.", "cybersecurity", "Evaluation", ["CY100"], "Discuss baseline activity profiling, anomaly thresholds, and privacy trade-offs."),
    
    # Additional CY queries up to 30
    ("What is an adversarial attack against machine learning malware classifiers?", "cybersecurity", "Definition", ["CY105"], "Define evasion attacks and feature perturbation in malware binaries."),
    ("How does DNS sinkholing mitigate botnet command and control traffic?", "cybersecurity", "Mechanism", ["CY110"], "Explain redirecting malicious domain queries to controlled isolation servers."),
    ("What are the limitations of air-gapped networks against sophisticated malware?", "cybersecurity", "Limitations", ["CY115"], "Discuss acoustic, electromagnetic, and USB supply chain bridge attacks."),
    ("How is federated learning used for privacy-preserving intrusion detection across banks?", "cybersecurity", "Applications", ["CY120"], "Discuss local model training and secure parameter aggregation."),
    ("How do security systems detect SQL injection attacks in web application firewalls?", "cybersecurity", "Detection", ["CY125"], "Discuss syntax tree parsing, regex matching, and input sanitization."),
    ("What are the challenges of securing serverless microservice architectures?", "cybersecurity", "Challenges", ["CY130"], "Discuss short lifecycle functions, broad permission scopes, and event spoofing."),
    ("Compare OAuth 2.0 and SAML authentication protocols.", "cybersecurity", "Comparison", ["CY135"], "Compare API authorization tokens vs XML-based enterprise SSO assertions."),
    ("How does machine learning predict cyber attack escalation patterns?", "cybersecurity", "Prediction", ["CY140"], "Discuss temporal attack graphs and state machine transition probabilities."),
    ("Evaluate the resilience of lattice-based post-quantum cryptography algorithms.", "cybersecurity", "Evaluation", ["CY145"], "Discuss resistance to Shor's algorithm and key size trade-offs.")
]

AG_QUESTIONS = [
    # Definition
    ("What is precision agriculture and how does site-specific management function?", "agriculture", "Definition", ["AG001", "AG002"], "Define precision farming, GPS mapping, and variable rate application."),
    ("What is climate-smart agriculture?", "agriculture", "Definition", ["AG005"], "Explain sustainable productivity, climate adaptation, and emissions reduction."),
    ("What is soil organic matter and why is it critical for crop health?", "agriculture", "Definition", ["AG010"], "Explain soil carbon, water retention, and microbial nutrient cycling."),
    
    # Mechanism
    ("How do deep learning convolutional networks identify crop foliar diseases from leaf images?", "agriculture", "Mechanism", ["AG015"], "Explain image feature extraction, lesion segmentation, and disease classification."),
    ("How does automated smart irrigation optimize water delivery based on soil moisture sensors?", "agriculture", "Mechanism", ["AG020"], "Explain sensor reading thresholds, weather forecasts, and automated valve actuation."),
    ("How do machine learning algorithms compute optimal crop selection recommendations?", "agriculture", "Mechanism", ["AG025"], "Explain analyzing soil NPK levels, rainfall, temperature, and yield potential."),
    
    # Advantages
    ("What are the main advantages of drone-based hyperspectral remote sensing in crop monitoring?", "agriculture", "Advantages", ["AG030"], "Discuss high spatial resolution, early stress detection, and rapid acreage coverage."),
    ("What are the benefits of conservation tillage in soil moisture preservation?", "agriculture", "Advantages", ["AG035"], "Discuss reduced soil erosion, enhanced organic carbon, and moisture retention."),
    ("What are the advantages of IoT sensor networks in greenhouse climate control?", "agriculture", "Advantages", ["AG040"], "Discuss real-time humidity/temp tracking, energy savings, and automated ventilation."),
    
    # Limitations / Challenges
    ("What are the main limitations of computer vision models deployed for weed detection in fields?", "agriculture", "Limitations", ["AG045"], "Discuss lighting variations, occlusion by crop canopy, and visual weed overlap."),
    ("What challenges hinder adoption of precision farming technologies among smallholder farmers?", "agriculture", "Challenges", ["AG050"], "Discuss high initial hardware costs, lack of digital connectivity, and technical literacy."),
    ("What are the challenges of predicting crop yields under extreme weather events?", "agriculture", "Challenges", ["AG055"], "Discuss nonlinear heat stress, flash droughts, and sparse localized weather data."),
    
    # Applications
    ("How is remote sensing applied to monitor global crop health using NDVI vegetation indices?", "agriculture", "Applications", ["AG060"], "Discuss near-infrared reflectance, chlorophyll absorption, and crop biomass tracking."),
    ("How are robotics and AI applied to selective automated fruit harvesting?", "agriculture", "Applications", ["AG065"], "Discuss 3D vision systems, soft robotic end-effectors, and ripeness classification."),
    ("How is machine learning used for soil nutrient analysis and fertilizer recommendation?", "agriculture", "Applications", ["AG070"], "Discuss spectroscopy data, NPK mapping, and site-specific fertilizer rates."),
    
    # Comparison
    ("Compare optical satellite imagery and UAV drone imagery for agricultural crop assessment.", "agriculture", "Comparison", ["AG075"], "Compare spatial resolution, flight operational costs, and revisit frequency."),
    ("Compare drip irrigation and sprinkler irrigation in water use efficiency.", "agriculture", "Comparison", ["AG080"], "Compare root-zone direct delivery vs evaporation losses."),
    ("Compare organic farming and conventional precision agriculture in environmental footprint.", "agriculture", "Comparison", ["AG085"], "Compare chemical input elimination vs yield per hectare efficiency."),
    
    # Detection / Prediction / Evaluation
    ("How can multispectral camera sensors detect plant water stress before visual wilting occurs?", "agriculture", "Detection", ["AG090"], "Discuss thermal canopy temperature elevation and stomatal closure signs."),
    ("How do machine learning models predict regional crop yield prior to harvest?", "agriculture", "Prediction", ["AG095"], "Discuss combining satellite NDVI time series, soil moisture, and weather metrics."),
    ("Evaluate the effectiveness of biological pest control compared to chemical pesticides.", "agriculture", "Evaluation", ["AG100"], "Discuss ecological balance, predator efficacy, and environmental safety."),
    
    # Additional AG queries up to 30
    ("What is vertical farming and how does hydroponic nutrient delivery work?", "agriculture", "Definition", ["AG105"], "Define indoor controlled environment agriculture and recirculating nutrients."),
    ("How do NIR spectroscopy sensors measure soil nitrogen content?", "agriculture", "Mechanism", ["AG110"], "Explain light absorption at specific NIR wavelengths correlated with organic N bonds."),
    ("What are the limitations of satellite NDVI in dense crop canopies?", "agriculture", "Limitations", ["AG115"], "Discuss spectral saturation at high leaf area index (LAI)."),
    ("How is artificial intelligence applied to automated livestock monitoring?", "agriculture", "Applications", ["AG120"], "Discuss computer vision tracking for cattle health, lameness, and feeding behavior."),
    ("How do machine learning models detect weed infestations in soybean crops?", "agriculture", "Detection", ["AG125"], "Discuss semantic segmentation networks distinguishing crop leaves from weed species."),
    ("What challenges affect autonomous agricultural tractor navigation in muddy terrain?", "agriculture", "Challenges", ["AG130"], "Discuss wheel slip, GPS signal degradation under canopy, and obstacle detection."),
    ("Compare hydroponic and aeroponic cultivation systems in water consumption.", "agriculture", "Comparison", ["AG135"], "Compare nutrient misting vs liquid nutrient bath efficiency."),
    ("How does machine learning predict frost damage risk in fruit orchards?", "agriculture", "Prediction", ["AG140"], "Discuss nocturnal microclimate temperature inversion forecasting."),
    ("Evaluate the impact of cover crops on soil health and weed suppression.", "agriculture", "Evaluation", ["AG145"], "Discuss nitrogen fixation, biomass mulch layer, and weed seed germination reduction.")
]

HC_QUESTIONS = [
    # Definition
    ("What is clinical decision support system in modern healthcare?", "healthcare", "Definition", ["HC001", "HC002"], "Define CDSS, EHR integration, and diagnostic alert mechanisms."),
    ("What is federated learning in healthcare research?", "healthcare", "Definition", ["HC005"], "Explain decentralized patient data training without centralizing sensitive records."),
    ("What is medical image segmentation?", "healthcare", "Definition", ["HC010"], "Explain delineating anatomical structures, tumors, and lesions in medical scans."),
    
    # Mechanism
    ("How do deep convolutional networks detect lung nodules in chest CT scans?", "healthcare", "Mechanism", ["HC015"], "Explain 3D convolution, candidate nodule detection, and false positive reduction."),
    ("How do medical natural language processing models extract clinical entities from physician notes?", "healthcare", "Mechanism", ["HC020"], "Explain bio-NER, assertion status detection, and medical ontology mapping."),
    ("How does explainable AI generate saliency maps for clinical radiologist verification?", "healthcare", "Mechanism", ["HC025"], "Explain Grad-CAM, attention visualization, and region-of-interest highlighting."),
    
    # Advantages
    ("What are the advantages of AI-assisted radiology in early cancer detection?", "healthcare", "Advantages", ["HC030"], "Discuss higher sensitivity for micro-calcifications, reduced reader fatigue, and speed."),
    ("What are the benefits of federated learning in multi-hospital collaborative research?", "healthcare", "Advantages", ["HC035"], "Discuss HIPAA compliance, data sovereignty, and larger multi-institutional cohorts."),
    ("What are the advantages of generative AI in de-novo drug molecule design?", "healthcare", "Advantages", ["HC040"], "Discuss rapid chemical space exploration, binding affinity optimization, and low cost."),
    
    # Limitations / Challenges
    ("What are the major limitations of deep learning models trained on single-site hospital datasets?", "healthcare", "Limitations", ["HC045"], "Discuss domain shift, scanner model bias, and poor out-of-distribution generalization."),
    ("What ethical and regulatory challenges affect clinical deployment of autonomous diagnostic AI?", "healthcare", "Challenges", ["HC050"], "Discuss medical liability, FDA software certification, and algorithmic bias."),
    ("What challenges hinder accurate clinical risk prediction from unstructured electronic health records?", "healthcare", "Challenges", ["HC055"], "Discuss missing values, noisy dictation text, and irregular temporal visits."),
    
    # Applications
    ("How is machine learning applied to early sepsis prediction in intensive care units?", "healthcare", "Applications", ["HC060"], "Discuss continuous vital sign streaming, physiological alert scores, and early intervention."),
    ("How is deep learning used in automated diabetic retinopathy screening from retinal fundus images?", "healthcare", "Applications", ["HC065"], "Discuss microaneurysm detection, exudate grading, and tele-ophthalmology screening."),
    ("How is natural language processing applied to clinical trial matching for oncology patients?", "healthcare", "Applications", ["HC070"], "Discuss extracting eligibility criteria, genomic markers, and patient EHR matching."),
    
    # Comparison
    ("Compare 2D CNNs and 3D vision transformers in volumetric MRI brain tumor segmentation.", "healthcare", "Comparison", ["HC075"], "Compare slice-by-slice processing vs full 3D spatial context encoding."),
    ("Compare rule-based expert systems and deep learning models in clinical decision support.", "healthcare", "Comparison", ["HC080"], "Compare explicit clinical guidelines vs data-driven pattern discovery."),
    ("Compare centralized medical data warehousing and federated learning in health privacy.", "healthcare", "Comparison", ["HC085"], "Compare central attack risks vs encrypted gradient communication."),
    
    # Detection / Prediction / Evaluation
    ("How do machine learning algorithms detect cardiac arrhythmias from wearable ECG signals?", "healthcare", "Detection", ["HC090"], "Discuss QRS complex detection, R-R interval variability, and temporal classification."),
    ("How do AI models predict 30-day hospital readmission risk for heart failure patients?", "healthcare", "Prediction", ["HC095"], "Discuss combining demographic factors, lab values, medication history, and past admissions."),
    ("Evaluate the clinical accuracy of automated mammography AI screening systems.", "healthcare", "Evaluation", ["HC100"], "Discuss sensitivity, specificity, recall rate, and false positive reduction."),
    
    # Additional HC queries up to 30
    ("What is BioBERT and how is it pre-trained for biomedical text mining?", "healthcare", "Definition", ["HC105"], "Define domain-specific BERT pre-trained on PubMed abstracts and PMC full text."),
    ("How does AlphaFold predict 3D protein structures from amino acid sequences?", "healthcare", "Mechanism", ["HC110"], "Explain evoformer blocks, pair representation, and spatial coordinate refinement."),
    ("What are the limitations of pulse oximetry AI algorithms in diverse skin tones?", "healthcare", "Limitations", ["HC115"], "Discuss optical absorption differences and systematic calibration bias."),
    ("How is computer vision applied to surgical phase recognition in laparoscopic procedures?", "healthcare", "Applications", ["HC120"], "Discuss video frame feature extraction and temporal state modeling."),
    ("How do deep learning models detect ischemic stroke lesions in non-contrast head CT?", "healthcare", "Detection", ["HC125"], "Discuss subtle hypodensity detection and Alberta Stroke Program Early CT Score."),
    ("What challenges affect the integration of AI models into hospital electronic health record systems?", "healthcare", "Challenges", ["HC130"], "Discuss HL7/FHIR interoperability, real-time latency, and clinical workflow friction."),
    ("Compare CT scans and MRI imaging modalities for deep learning stroke classification.", "healthcare", "Comparison", ["HC135"], "Compare bone contrast / acquisition speed vs soft tissue resolution."),
    ("How does AI predict acute kidney injury onset 48 hours prior to clinical diagnosis?", "healthcare", "Prediction", ["HC140"], "Discuss longitudinal serum creatinine, urine output, and vital sign gradient models."),
    ("Evaluate the diagnostic performance of AI chatbots in triage symptom assessment.", "healthcare", "Evaluation", ["HC145"], "Discuss safety thresholds, misdiagnosis risk, and concordance with physician triage.")
]

CL_QUESTIONS = [
    # Definition
    ("What is climate modeling and how do Earth System Models operate?", "climate", "Definition", ["CL001", "CL002"], "Define ESMs, atmosphere-ocean fluid dynamics, and thermodynamic coupling."),
    ("What is equilibrium climate sensitivity?", "climate", "Definition", ["CL005"], "Explain global mean surface temperature change resulting from doubled atmospheric CO2."),
    ("What is greenhouse gas flux in atmospheric monitoring?", "climate", "Definition", ["CL010"], "Explain exchange rate of carbon dioxide and methane between surface and atmosphere."),
    
    # Mechanism
    ("How do deep learning atmospheric emulators accelerate global weather forecasting?", "climate", "Mechanism", ["CL015"], "Explain neural operators, spherical mesh convolutions, and autoregressive stepping."),
    ("How does remote sensing satellite spectroscopy measure atmospheric methane emissions?", "climate", "Mechanism", ["CL020"], "Explain shortwave infrared solar backscatter absorption at specific gas bands."),
    ("How do climate models simulate cloud feedback mechanisms under global warming?", "climate", "Mechanism", ["CL025"], "Explain cloud optical depth, cloud fraction parameterization, and thermal radiative forcing."),
    
    # Advantages
    ("What are the advantages of AI surrogate models over traditional numerical weather prediction?", "climate", "Advantages", ["CL030"], "Discuss orders of magnitude faster execution time, reduced supercomputing cost, and scalability."),
    ("What are the benefits of high-resolution satellite remote sensing in forest carbon stock estimation?", "climate", "Advantages", ["CL035"], "Discuss wall-to-wall global coverage, canopy height LiDAR metrics, and temporal tracking."),
    ("What are the advantages of physics-informed neural networks in ocean dynamics modeling?", "climate", "Advantages", ["CL040"], "Discuss enforcing mass/energy conservation equations within loss functions."),
    
    # Limitations / Challenges
    ("What are the major limitations of current machine learning weather forecasting models for extreme events?", "climate", "Limitations", ["CL045"], "Discuss blurring of out-of-distribution extremes and lack of physical conservation guarantees."),
    ("What challenges hinder accurate long-term precipitation forecasting in climate models?", "climate", "Challenges", ["CL050"], "Discuss sub-grid convection scale, complex aerosol-cloud microphysics, and topography."),
    ("What are the primary challenges in quantifying urban greenhouse gas emission inventories?", "climate", "Challenges", ["CL055"], "Discuss point-source plume dispersion, background atmospheric noise, and sparse sensors."),
    
    # Applications
    ("How is machine learning applied to tropical cyclone intensity and track prediction?", "climate", "Applications", ["CL060"], "Discuss satellite infrared imagery analysis, sea surface temperature, and pressure gradients."),
    ("How are deep learning models used for automated flood risk mapping and inundation prediction?", "climate", "Applications", ["CL065"], "Discuss synthetic aperture radar (SAR) imagery, elevation DEMs, and hydrological flow."),
    ("How is remote sensing applied to monitor polar ice sheet melting and sea level rise?", "climate", "Applications", ["CL070"], "Discuss satellite radar altimetry, ice velocity tracking, and grounding line retreat."),
    
    # Comparison
    ("Compare global climate models (GCMs) and regional climate downscaling models (RCMs).", "climate", "Comparison", ["CL075"], "Compare coarse global grid resolution vs fine-scale regional topographical detail."),
    ("Compare machine learning weather forecasting (FourCastNet/Pangu-Weather) and numerical NWP.", "climate", "Comparison", ["CL080"], "Compare data-driven inference speed vs physical partial differential equation solvers."),
    ("Compare optical remote sensing and microwave Synthetic Aperture Radar (SAR) for flood monitoring.", "climate", "Comparison", ["CL085"], "Compare sunlight dependence vs cloud-penetrating night/day SAR imaging."),
    
    # Detection / Prediction / Evaluation
    ("How do satellite algorithms detect methane super-emitter plumes from industrial facilities?", "climate", "Detection", ["CL090"], "Discuss spectral matched filter analysis and column enhancement ratio detection."),
    ("How do climate ML models predict sub-seasonal to seasonal drought conditions?", "climate", "Prediction", ["CL095"], "Discuss soil moisture deficit indices, El Nino Southern Oscillation (ENSO) anomalies."),
    ("Evaluate the accuracy of machine learning models in predicting heatwave duration and peak temperatures.", "climate", "Evaluation", ["CL100"], "Discuss mean absolute error, peak temperature underestimation, and lead time metrics."),
    
    # Additional CL queries up to 30
    ("What is atmospheric aerosol radiative forcing in climate dynamics?", "climate", "Definition", ["CL105"], "Define aerosol light scattering and absorption altering Earth's energy balance."),
    ("How do neural network surrogates emulate atmospheric chemistry in Earth System Models?", "climate", "Mechanism", ["CL110"], "Explain approximating stiff chemical reaction differential equations with deep neural nets."),
    ("What are the limitations of satellite radar altimetry in coastal sea level measurement?", "climate", "Limitations", ["CL115"], "Discuss land contamination of radar footprints near coastlines."),
    ("How is machine learning applied to renewable wind and solar power generation forecasting?", "climate", "Applications", ["CL120"], "Discuss numerical weather prediction inputs, cloud movement tracking, and grid balancing."),
    ("How do deep learning models detect wildfire ignition and burn scar extent from satellite images?", "climate", "Detection", ["CL125"], "Discuss active thermal hotspot detection and post-fire NDVI spectral burn ratios."),
    ("What challenges affect modeling ocean overturning circulation under ice sheet melting scenarios?", "climate", "Challenges", ["CL130"], "Discuss freshwater hosing feedback, brine rejection physics, and decadal simulation lengths."),
    ("Compare carbon capture efficiency metrics across direct air capture and point-source capture.", "climate", "Comparison", ["CL135"], "Compare low ambient CO2 concentration energetic cost vs high-purity flue gas energy."),
    ("How does machine learning predict sea surface temperature anomalies during El Nino events?", "climate", "Prediction", ["CL140"], "Discuss spatio-temporal graph convolutional networks modeling Pacific trade winds."),
    ("Evaluate the performance of machine learning downscaling for extreme urban heat island mapping.", "climate", "Evaluation", ["CL145"], "Discuss spatial resolution improvements, land surface temperature accuracy, and canopy bias.")
]

MULTI_DOMAIN_QUESTIONS = [
    ("How is artificial intelligence applied in healthcare medical imaging to assist clinical diagnosis?", ["artificial_intelligence", "healthcare"], "Applications", ["AI001", "HC001"], "Requires evidence from both AI and Healthcare domains."),
    ("How does smart agriculture utilize AI-based computer vision for crop disease detection?", ["agriculture", "artificial_intelligence"], "Mechanism", ["AG001", "AI002"], "Requires evidence from both Agriculture and AI domains."),
    ("How do climate change weather predictions impact agricultural crop yield forecasting?", ["climate", "agriculture"], "Effect", ["CL001", "AG002"], "Requires evidence from both Climate and Agriculture domains."),
    ("How is machine learning used to enhance cybersecurity threat detection in healthcare medical devices?", ["cybersecurity", "healthcare"], "Applications", ["CY001", "HC002"], "Requires evidence from both Cybersecurity and Healthcare domains."),
    ("How does AI-driven remote sensing help evaluate climate change impacts on forest carbon sinks?", ["artificial_intelligence", "climate"], "Applications", ["AI005", "CL002"], "Requires evidence from both AI and Climate domains."),
    ("How are IoT sensors used in precision agriculture for climate-resilient water management?", ["agriculture", "climate"], "Applications", ["AG005", "CL005"], "Requires evidence from both Agriculture and Climate domains."),
    ("How do federated machine learning methods preserve patient privacy in multi-hospital healthcare AI?", ["healthcare", "cybersecurity", "artificial_intelligence"], "Mechanism", ["HC005", "CY010", "AI025"], "Requires cross-domain evidence from Healthcare, Cybersecurity, and AI."),
    ("How are explainable AI models deployed in clinical decision support for oncology diagnosis?", ["artificial_intelligence", "healthcare"], "Applications", ["AI005", "HC001"], "Requires cross-domain evidence from AI and Healthcare."),
    ("How do autonomous agricultural drones integrate computer vision for real-time pest detection?", ["agriculture", "artificial_intelligence"], "Mechanism", ["AG015", "AI035"], "Requires evidence from Agriculture and AI."),
    ("How does cybersecurity privacy-preserving homomorphic encryption safeguard cloud-based climate data?", ["cybersecurity", "climate"], "Applications", ["CY010", "CL010"], "Requires evidence from Cybersecurity and Climate.")
]

OUT_OF_CORPUS_QUESTIONS = [
    ("What is surface code quantum error correction?", "artificial_intelligence", "Definition", [], "Out-of-corpus: Must trigger ArXiv online academic fallback [O1][O2]."),
    ("How do topological qudit decoders compute stabilizer syndromes in fault-tolerant quantum computing?", "artificial_intelligence", "Mechanism", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("What is room-temperature superconductivity in ambient pressure hydride compounds?", "climate", "Definition", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("How do Mamba state space models compare to Transformers in long-context sequence modeling?", "artificial_intelligence", "Comparison", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("What are the key mechanisms of CRISPR-Cas13 RNA editing in viral disease therapeutics?", "healthcare", "Mechanism", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("How does Neuromorphic Computing implement spiking neural hardware for event camera processing?", "artificial_intelligence", "Mechanism", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("What are zero-knowledge succinct non-interactive arguments of knowledge (zk-SNARKs) in blockchain privacy?", "cybersecurity", "Definition", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("How do peristaltic micropumps function in organ-on-a-chip microfluidic drug screening?", "healthcare", "Mechanism", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("What is gravitational wave interferometry detection using pulsar timing arrays?", "climate", "Definition", [], "Out-of-corpus: Must trigger ArXiv online academic fallback."),
    ("How do post-quantum lattice-based Kyber and Dilithium algorithms resist quantum decryption?", "cybersecurity", "Mechanism", [], "Out-of-corpus: Must trigger ArXiv online academic fallback.")
]


def build_benchmark_dataset():
    benchmark_items = []
    qid = 1

    for q_text, dom, intent, pids, criteria in AI_QUESTIONS:
        benchmark_items.append({
            "id": f"BENCH-{qid:03d}",
            "question": q_text,
            "expected_domain": dom,
            "expected_intent": intent,
            "relevant_paper_ids": pids,
            "expected_evidence_criteria": criteria,
            "category": "single_domain"
        })
        qid += 1

    for q_text, dom, intent, pids, criteria in CY_QUESTIONS:
        benchmark_items.append({
            "id": f"BENCH-{qid:03d}",
            "question": q_text,
            "expected_domain": dom,
            "expected_intent": intent,
            "relevant_paper_ids": pids,
            "expected_evidence_criteria": criteria,
            "category": "single_domain"
        })
        qid += 1

    for q_text, dom, intent, pids, criteria in AG_QUESTIONS:
        benchmark_items.append({
            "id": f"BENCH-{qid:03d}",
            "question": q_text,
            "expected_domain": dom,
            "expected_intent": intent,
            "relevant_paper_ids": pids,
            "expected_evidence_criteria": criteria,
            "category": "single_domain"
        })
        qid += 1

    for q_text, dom, intent, pids, criteria in HC_QUESTIONS:
        benchmark_items.append({
            "id": f"BENCH-{qid:03d}",
            "question": q_text,
            "expected_domain": dom,
            "expected_intent": intent,
            "relevant_paper_ids": pids,
            "expected_evidence_criteria": criteria,
            "category": "single_domain"
        })
        qid += 1

    for q_text, dom, intent, pids, criteria in CL_QUESTIONS:
        benchmark_items.append({
            "id": f"BENCH-{qid:03d}",
            "question": q_text,
            "expected_domain": dom,
            "expected_intent": intent,
            "relevant_paper_ids": pids,
            "expected_evidence_criteria": criteria,
            "category": "single_domain"
        })
        qid += 1

    for q_text, doms, intent, pids, criteria in MULTI_DOMAIN_QUESTIONS:
        benchmark_items.append({
            "id": f"BENCH-{qid:03d}",
            "question": q_text,
            "expected_domain": doms,
            "expected_intent": intent,
            "relevant_paper_ids": pids,
            "expected_evidence_criteria": criteria,
            "category": "multi_domain"
        })
        qid += 1

    for q_text, dom, intent, pids, criteria in OUT_OF_CORPUS_QUESTIONS:
        benchmark_items.append({
            "id": f"BENCH-{qid:03d}",
            "question": q_text,
            "expected_domain": dom,
            "expected_intent": intent,
            "relevant_paper_ids": pids,
            "expected_evidence_criteria": criteria,
            "category": "online_fallback"
        })
        qid += 1

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_items, f, indent=2)

    print("==========================================================================")
    print(f"=== RESEARCH MIND EVALUATION BENCHMARK DATASET GENERATED ===")
    print("==========================================================================")
    print(f"Total benchmark questions: {len(benchmark_items)}")
    print(f"  Single Domain AI: 30")
    print(f"  Single Domain Cybersecurity: 30")
    print(f"  Single Domain Agriculture: 30")
    print(f"  Single Domain Healthcare: 30")
    print(f"  Single Domain Climate: 30")
    print(f"  Multi-Domain Queries: 10")
    print(f"  Out-of-Corpus / Online Fallback Queries: 10")
    print(f"Saved benchmark to: {BENCHMARK_PATH}")
    print("==========================================================================")


if __name__ == "__main__":
    build_benchmark_dataset()
