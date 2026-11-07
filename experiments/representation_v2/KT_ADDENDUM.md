# Beginner KT addendum: representation v2

7 November 2026, Asia/Karachi. Yeh **alag experiment** hai. Purani 30-chapter KT guide baseline ka dated snapshot hai aur usay badla nahi gaya.

## Kya result mila?

Validation se select hua model **SVM**, key **fusion-13** hai. Purane SVM ki validation accuracy 50.17% thi; naye winner ki **51.40%** hai. Macro-F1 0.4983 se **0.5091** hua. Accuracy ka farq **+1.22 percentage points** hai. Percentage points seedha do percentages ka subtraction hota hai; isay relative percent increase ke saath mix nahi karna.

Purane selected SVM ki test accuracy 55.26% thi; naye selected winner ki **57.72%** hai. Yeh **reused test** hai: hum baseline ke test results pehle dekh chuke thay. Is liye isay bilkul naya untouched test kehna ghalat hoga. Naya winner validation se freeze kiya gaya, test dekh kar badla nahi gaya. Agar kisi doosre model ka test score behtar ho, tab bhi winner automatically change nahi hota.

## Features ko kyun badla?

Baseline har clip ko 273 numbers mein badalta tha. Bohat se numbers frames ka mean/std thay. Mean tumhein average batata hai, lekin pitch kab upar gayi aur voice kitni unstable thi uski poori detail nahi deta. openSMILE eGeMAPSv02 speech research ka established 88-feature set hai. Yeh recording ko measure karta hai; internet se trained emotion labels wala model download nahi hua.

- **F0 / pitch:** awaaz ki bunyadi frequency. Semitone representation, percentiles aur rising/falling slopes awaaz ka range aur movement summarize karte hain. Pitch emotion ki guarantee nahi; speaker bhi pitch ko affect karta hai.
- **Loudness:** perceptual energy ka estimate. Percentiles distribution ko samjhate hain; peaks per second rough rhythm cue dete hain.
- **Jitter / shimmer:** pitch periods aur amplitude ki short-term irregularity ke estimates. Noisy recording mein estimates imperfect ho sakte hain; medical diagnosis nahi hain.
- **HNR:** harmonic aur noise-like components ka ratio estimate. Yeh clean paired reference se measured denoising SNR improvement nahi hai.
- **Formants:** vocal tract ke resonance bands, roughly F1/F2/F3; voice production ki information dete hain.
- **Voiced/unvoiced segments:** voice vibration wale aur baghair vibration wale hisson ki lengths aur counts. Yeh transcript ya sentence meaning nahi dete.

`egemaps` branch mein 88 numbers unfiltered audio se aate hain. `fusion` mein baseline ke 273 filtered numbers aur 88 raw-audio numbers jurtay hain: **273 + 88 = 361**. Low-pass ka existing implementation use hota hai. Raw supplementary features ko filtered keh kar present nahi kiya. eGeMAPS bhi akhir mein clip summaries banata hai, is liye temporal information ka masla poori tarah solve hone ka claim nahi.

## Grouped validation kaise kaam karti hai?

Original training mein 63 actors hain. Unhi actors ko 3 internal folds mein baanta. Har round mein kuch training actors fit ke liye aur alag actors internal holdout ke liye thay. Ek actor ki clips dono taraf nahi gayin. Original 14 validation aur 14 test actors internal search mein shamil nahi thay. Is se hyperparameters ko naye speakers par compare karna random clips ke muqable mein munasib hai.

Har representation par KNN, Logistic Regression, GaussianNB, SVM aur MLP mila kar 20 configurations thay. Dono representations se 40 configurations, har ek ke 3 fits: **120 internal fits**. Har family/representation ka best configuration training CV se chuna, phir 10 finalists ko poore original training par fit kiya. Original validation ne final winner select kiya. Validation par scaler ya classifier fit nahi hua.

StandardScaler ke mean/std sirf fit actors se seekhay gaye. Optional shrinkage LDA label-aware projection hai: covariance estimates ko stable banane ke liye shrinkage lagti hai. Isay har fold ke andar fit karna zaroori hai, warna holdout information leak ho sakti hai. MLP mein random-clip early stopping disable hai. Agar convergence warning ho to iska matlab optimizer apne stopping criterion tak nahi pohncha; model ki predictions phir bhi available hoti hain aur warning CSV mein rehti hai.

## Is result ka kitna bharosa karein?

Macro-F1 har emotion ke F1 ko barabar weight deta hai. Accuracy sirf total correct fraction hai. Dono dekhna behtar hai. Training accuracy **83.03%** aur validation **51.40%** ka gap ab bhi generalization ki limitation dikha sakta hai.

Actor bootstrap mein poora actor resample hota hai, individual clips nahi. Ek actor ki clips related hoti hain, is liye clip bootstrap uncertainty ko chhota dikha sakta hai. Paired comparison mein purana aur naya model bilkul wohi sampled actors dekhte hain. Is se score difference ka descriptive interval milta hai. Lekin yeh training aur winner selection ki uncertainty cover nahi karta. Validation pehle baseline mein bhi use hui thi aur ab winner select karti hai; is liye independent final proof nahi. Naye actors ka external corpus strong next check hoga.

Representation aur regularization/search dono badlay hain. Is experiment se yeh nahi keh sakte ke poora gain sirf ek particular pitch feature ki wajah se hua. 80-90% ki guarantee nahi. Noise filtering se clean SNR improve hua, yeh bhi measure nahi kiya.

## PyCharm mein chalana aur inference

Project `D:\ML` open karo, interpreter `.venv\Scripts\python.exe` select karo. Isi folder ka `improvement.ipynb` kholo aur Run All karo. Verified caches milne par expensive extraction/training repeat nahi hogi; identity mismatch aaye to usay bypass mat karo. `PROTOCOL.md` mein commands aur exact search hai. `comparison_report.md` measured tables deta hai. `features_metadata.json` mein tamam 88 feature names hain.

```python
from experiments.representation_v2.features import predict_emotion
print(predict_emotion('path/to/recording.wav'))
```

Inference wohi feature recipe, training scaler, optional LDA aur saved classifier use karta hai. Baseline inference ab bhi `src.inference.predict_emotion` hai. Dono functions ko confuse mat karo. Naya result baseline artifact ko replace nahi karta. Sirf trusted apne joblib files load karo.

## Files aur next handoff

`features.py` per-recording features aur inference, `run.py` grouped search/freeze/test, `verify.py` leakage aur real-artifact checks, `deliver.py` report/notebook banata hai. `selection.json` ki key final choice hai. `baseline_hashes.json` original files ki identities rakhta hai. New requirements-lock.txt extended environment record karta hai; purana lock unchanged hai. Dataset aur original notebook/report ki allocations preserve hain.

Agli chat ko batao: **v2 complete results isi folder mein hain; pehle selection.json aur comparison_report.md parho, test ko tune karne ke liye dobara use mat karo, aur baseline files mat badlo.** Naya serious experiment alag folder mein ho. Koi commit/push/publication authorize nahi ki gayi. openSMILE educational/research use ke liye free hai; commercial product ki licensing alag hai. References aur Codex assistance ka attribution PROTOCOL.md mein hai.

## Aapki apni voice ka test

Aapki confirmation ke mutabiq 5 happy, 5 disgust, 5 sad aur 5 angry, total20 real recordings ka desktop recorder bhi hai. `voice_evaluation/README.md` mein instructions hain. Mic sirf Record dabane se active hota hai. Tamam clips pehle collect hoti hain; uske baad frozen baseline aur improved model predict karte hain. Recordings training ya hyperparameter tuning mein use nahi hotin. Fear/neutral output aaye to error count hota hai. Chaar supported emotions ka F1 aur six-fixed-label F1 alag hain. Ek shakhs ki 20 takes ko 20 independent speakers mat samajhna. Real personal results comparison report mein alag section mein hain; incomplete session ki accuracy invent nahi hoti.

## Scope: asal CEP aur extra kaam alag hain

Asal CEP Q2 ka maqsad assigned audio se emotions classify karna hai: pehle split, phir low-pass filtering aur justified features, phir five required classifiers aur evaluation. Purana verified notebook aur baseline report yeh core evidence hain. Abhi humne jo eGeMAPS/fusion research aur aapki microphone ki 20 takes ki hain woh **aapki request par extra kaam** hai; teacher ki extra mandatory requirement keh kar present nahi karna. Main report mein core pehle, yeh extensions appendices mein hain.

### Har experiment successful nahi tha

eGeMAPS-only ke best finalist ki validation macro-F1 0.4491 thi, jo baseline 0.4983 se kam hai. Fusion SVM ne chhota point gain diya. Bootstrap intervals zero ko cross karte hain, is liye statistically pakki superiority claim nahi. WavLM implement/train nahi hua. Temporal information ka masla poori tarah solve nahi hua. No clean-reference SNR improvement measure ki gayi.

### Aapki real voice par kya hua?

Aapne khud 20 real recordings ki: har requested emotion ki five takes. Dono models 5/20 yani **25%** correct thay. Naya model 19 clips ko happy aur ek ko fear keh raha tha. Usne sad, disgust aur angry ki koi clip sahi nahi ki. Is liye microphone use par improvement ka claim ghalat hoga. Four-supported-class macro-F1 baseline 0.1801 aur v2 0.1042 hai. Accuracy same hone ka matlab dono equally useful nahi; class-wise predictions bhi dekhni hoti hain.

Constant happy kehne wala model bhi is balanced session mein 25% le leta. Yeh real-use limitation ka saboot hai. Recording peaks/clipping checks theek hona emotion generalization ki guarantee nahi. Mic/accent/room/acting aur silence distribution possible reasons hain; actual causal test nahi kiya, is liye kisi ek ko pakki wajah nahi kehte. Scores dekh kar recordings trim, replace ya model tune nahi kiya. Future adaptation karni ho to in takes ko development data kehna hoga aur alag fresh evaluation recordings collect karni hongi.

| filename | intended_emotion | baseline_prediction | improved_prediction |
| --- | --- | --- | --- |
| 01_anger.wav | anger | happy | happy |
| 02_sad.wav | sad | fear | fear |
| 03_disgust.wav | disgust | disgust | happy |
| 04_happy.wav | happy | disgust | happy |
| 05_anger.wav | anger | happy | happy |
| 06_sad.wav | sad | disgust | happy |
| 07_happy.wav | happy | happy | happy |
| 08_disgust.wav | disgust | disgust | happy |
| 09_happy.wav | happy | fear | happy |
| 10_anger.wav | anger | disgust | happy |
| 11_sad.wav | sad | fear | happy |
| 12_disgust.wav | disgust | fear | happy |
| 13_sad.wav | sad | fear | happy |
| 14_anger.wav | anger | disgust | happy |
| 15_happy.wav | happy | disgust | happy |
| 16_disgust.wav | disgust | disgust | happy |
| 17_sad.wav | sad | disgust | happy |
| 18_anger.wav | anger | disgust | happy |
| 19_happy.wav | happy | disgust | happy |
| 20_disgust.wav | disgust | disgust | happy |

Voice data local ignored sessions folder mein hai. Dono models six labels predict karte hain; fear/neutral ki true clips is test mein nahi hain, lekin unki predictions errors hoti hain. Yeh one-speaker personal test hai, multi-speaker validation nahi. Complete per-class tables, metrics aur caveats combined report mein hain.

## Extra investigation: kya features ghalat thay? (v3)

Is sawal ko sirf guess se answer nahi kiya. Training data mein 794 highly correlated feature pairs mili. Phir har training fold ke andar ANOVA feature selection lagayi: 32,64,128,256 ya poore361 features. 35 configurations aur3actor-grouped folds se105 internal fits hue. Logistic Regression aur SVM dono ke liye stronger regularization bhi test hui. Core CEP ke five required classifiers pehle se maujood hain; yeh two-family test alag diagnostic hai.

**Actual result: dono families ne poore361 features select kiye.** Reduced feature subsets ke grouped CV scores kam thay. Selected SVM phir wohi v2 configuration bani: validation51.40%, macro-F10.5091. Naya gain0. Is liye “features kam kar diye aur accuracy improve ho gayi” likhna ghalat hoga. Test aur aapki20 voice clips ko is experiment mein dobara evaluate nahi kiya.

Correlation ka matlab milti-julti information hai; yeh automatically bekaar feature ya overfitting ki pakki wajah nahi. Univariate ANOVA har feature ko alag rank karti hai; useful interactions miss ho sakti hain. Is test se sirf itna pata chala ke tested selection strategy se gain nahi mila. Har possible selection method fail hoga, yeh prove nahi hua.

Overfitting ko training/validation gap se pehchante hain, magar gap ki exact wajah ke liye controlled tests chahiye. 5149training clips sirf63actors se aati hain; actor-related patterns independent new speakers par fail ho sakti hain. Personal microphone recording ka domain bhi dataset se different hai. Voice par25% result ko sirf “features ghalat” ya “aapne emotion sahi nahi bola” keh kar explain nahi kar sakte. Evidence itna strong nahi.

Agla scientific direction stronger temporal/pretrained speech representation ya carefully designed domain-robust training ho sakti hai. Woh abhi implement/train nahi hui; RAM bhi limited hai. Maujooda20recordings par tune karenge to unhein development data kehna hoga aur fresh recordings se confirmation chahiye. Core CEP ka kaam aur optional experiments ki limitations report mein alag hain.

## Reference project ko kaise compare karein

# Reference repository: reported performance and evaluation caveats

Reference: Abdullahnaveed102/Sentiment-Analysis-of-Audio-Recordings, inspected at commit `69bc3cc3a10d59703eccd3cc04fe6ed308720cba`. Yeh source/report review hai; upstream notebook hamari machine par execute nahi ki gayi. Original CEP implementation independently authored hai.

| Classifier | Saved notebook accuracy | PDF reported accuracy |
| --- | --- | --- |
| KNN | 54.16% | 54.1% |
| Logistic Regression | 55.67% | 55.4% |
| Gaussian Naive Bayes | 52.98% | 55.2% |
| SVM | 55.51% | 55.6% |
| Neural Network | 55.35% | 54.9% |

Saved notebook ka best Logistic Regression hai; PDF SVM ko best kehti hai. Numbers consistent nahi hain. Is farq ki exact wajah published material se establish nahi hui. Reference bhi roughly55-56% hai; 80-90% result published nahi hai.

Sources: [saved notebook](https://github.com/Abdullahnaveed102/Sentiment-Analysis-of-Audio-Recordings/blob/69bc3cc3a10d59703eccd3cc04fe6ed308720cba/Mian_NoteBook.ipynb), [PDF report](https://github.com/Abdullahnaveed102/Sentiment-Analysis-of-Audio-Recordings/blob/69bc3cc3a10d59703eccd3cc04fe6ed308720cba/Final_Report.pdf). Source file identities and URLs are archived in provenance.json.

Notebook code mein `lda.fit_transform(X_normal, y)` **train_test_split se pehle** poore7442examples aur unke labels par chalti hai. LDA supervised hai, is liye test-label information representation mein leak hoti hai. Baad mein random clip-level75/25split hai (`test_size=0.25`, seed7), actor grouping enforce nahi. Separate validation stage nazar nahi aati. F1 `average='micro'` hai, jo is single-label task mein accuracy ke barabar hai; yeh macro-F1 nahi.

PDF80/20split aur2kHzcutoff describe karti hai, jabke saved notebook75/25aur5kHz use karti hai. Yeh reproducibility discrepancies hain. Hamare actor-disjoint, training-only preprocessing protocol ke saath direct numerical superiority claim fair nahi. Hamara baseline55.26% aur optional v2reused-test57.72% apni allocation par measured values hain; reference ke scores unki published outputs hain. Personal microphone result25% ek alag one-speaker test hai. Kisi score ko doosre evaluation design ka substitute na banayein.
