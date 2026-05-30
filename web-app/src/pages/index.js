import { useEffect, useRef, useState } from 'react';
import Head from 'next/head';
import * as ort from 'onnxruntime-web';

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

const MAIN_CONTROLS = [
  { key: 'cement', label: 'Cement (kg/m³)', min: 100, max: 600, step: 1 },
  { key: 'TCM', label: 'Total Cementitious Material (kg/m³)', min: 100, max: 800, step: 1 },
  { key: 'water', label: 'Water (kg/m³)', min: 100, max: 250, step: 1 },
  { key: 'water_TCM', label: 'Water–TCM Ratio', min: 0.2, max: 0.8, step: 0.01 },
  { key: 'AGE', label: 'Curing Age (days)', min: 1, max: 365, step: 1 },
];

const ADVANCED_TABS = {
  "Binders": [
    { key: 'flyash', label: 'Fly Ash (kg/m³)', min: 0, max: 300, step: 1 },
    { key: 'GGBS', label: 'Ground Granulated Blast Furnace Slag (kg/m³)', min: 0, max: 300, step: 1 },
    { key: 'MK', label: 'Metakaolin (kg/m³)', min: 0, max: 100, step: 1 },
  ],
  "Aggregates": [
    { key: 'SAND', label: 'Sand (kg/m³)', min: 300, max: 1000, step: 1 },
    { key: 'NCA_20_DOWN', label: 'Natural Coarse Aggregate (20 mm) (kg/m³)', min: 0, max: 1000, step: 1 },
    { key: 'NCA_10_DOWN', label: 'Natural Coarse Aggregate (10 mm) (kg/m³)', min: 0, max: 1000, step: 1 },
    { key: 'RCA_20_Down', label: 'Recycled Coarse Aggregate (20 mm) (kg/m³)', min: 0, max: 1000, step: 1 },
    { key: 'RCA_10DOWN', label: 'Recycled Coarse Aggregate (10 mm) (kg/m³)', min: 0, max: 1000, step: 1 },
  ],
  "Admixtures": [
    { key: 'SP', label: 'Superplasticizer Percentage (%)', min: 0, max: 15, step: 0.1 },
    { key: 'VMA', label: 'Viscosity Modifying Agent Percentage (%)', min: 0, max: 5, step: 0.1 },
  ]
};

const FEATURE_ORDER = [
  'cement',
  'flyash',
  'GGBS',
  'MK',
  'TCM',
  'water',
  'water_TCM',
  'SP',
  'VMA',
  'NCA_20_DOWN',
  'NCA_10_DOWN',
  'RCA_20_Down',
  'RCA_10DOWN',
  'SAND',
  'AGE',
];


export default function Home() {
   const [prediction, setPrediction] = useState(null);
   const [modelReady, setModelReady] = useState(false);
   const [loadingProgress, setLoadingProgress] = useState(0);
   const sessionRef = useRef(null);
   const isRunningRef = useRef(false);
   const pendingRef = useRef(false);
   const categoryNames = Object.keys(ADVANCED_TABS);
   const [activeCategory, setActiveCategory] = useState(categoryNames[0]);
   const [activeSection, setActiveSection] = useState('mix');

  const [formData, setFormData] = useState({
    cement: 350, flyash: 0, GGBS: 0, MK: 0, TCM: 350, water: 160,
    water_TCM: 0.46, SP: 2.5, VMA: 0, NCA_20_DOWN: 500, NCA_10_DOWN: 300,
    RCA_20_Down: 0, RCA_10DOWN: 0, SAND: 700, AGE: 28
  });

  const formatNumber = (value) => {
    const fixed = parseFloat(value.toFixed(2));
    return fixed;
  };

  const handleChange = (e) => {
    const name = e.target.name;
    const nextValue = parseFloat(e.target.value) || 0;
    
    setFormData((prev) => {
      const updated = { ...prev, [name]: nextValue };
      
      if (name === 'water' || name === 'TCM') {
        const tcm = name === 'TCM' ? nextValue : updated.TCM;
        updated.water_TCM = tcm > 0 ? updated.water / tcm : 0;
        updated.water_TCM = Math.round(updated.water_TCM * 100) / 100;
      }
      
      if (name === 'water_TCM') {
        updated.water_TCM = Math.round(nextValue * 100) / 100;
        const calculatedWater = updated.water_TCM * updated.TCM;
        
        if (calculatedWater > 250) {
          updated.TCM = Math.round(updated.water / updated.water_TCM * 100) / 100;
        } else {
          updated.water = Math.round(calculatedWater * 100) / 100;
        }
      }
      
      return updated;
    });
  };

   useEffect(() => {
     let isMounted = true;
     const loadSession = async () => {
       try {
         // Simulate realistic progress that takes time to reach 100%
         const progressInterval = setInterval(() => {
           setLoadingProgress((prev) => {
             if (prev < 30) {
               return prev + Math.random() * 15; // Slow start
             } else if (prev < 60) {
               return prev + Math.random() * 8; // Medium pace
             } else if (prev < 90) {
               return prev + Math.random() * 5; // Slower as we approach
             } else {
               return prev; // Stop before 100%
             }
           });
         }, 400);

         const session = await ort.InferenceSession.create('/concrete_model.onnx');
         clearInterval(progressInterval);
         
         if (!isMounted) return;
         sessionRef.current = session;
         // Jump to 100% when actually ready
         setLoadingProgress(100);
         
         // Immediately show the app once loading is done
         setTimeout(() => {
           if (isMounted) {
             setModelReady(true);
           }
         }, 500);
       } catch (error) {
         console.error('Model Load Error:', error);
         setLoadingProgress(0);
       }
     };
     loadSession();
     return () => { isMounted = false; };
   }, []);

  const runInference = async () => {
    if (!modelReady || !sessionRef.current) return;
    if (isRunningRef.current) {
      pendingRef.current = true;
      return;
    }
    
    isRunningRef.current = true;
    try {
      const inputValues = FEATURE_ORDER.map((key) => formData[key]);
      const inputTensor = new ort.Tensor('float32', Float32Array.from(inputValues), [1, 15]);
      const feeds = { [sessionRef.current.inputNames[0]]: inputTensor };
      const results = await sessionRef.current.run(feeds);
      setPrediction(results[sessionRef.current.outputNames[0]].data[0]);
    } catch (error) {
      console.error('Inference Error:', error);
    } finally {
      isRunningRef.current = false;
      if (pendingRef.current) {
        pendingRef.current = false;
        runInference();
      }
    }
  };

  useEffect(() => {
    if (!modelReady) return;
    runInference();
  }, [formData, modelReady]);

  const MAX_MPA = 120;
  const strengthPercentage = prediction ? Math.min((prediction / MAX_MPA) * 100, 100) : 0;

  return (
    <>
      <Head>
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap"
        />
        <title>Concrete Strength Estimator</title>
        <style>{`
           * { box-sizing: border-box; }
           body { 
             margin: 0; 
             background-color: #000000; 
             color: #f8fafc; 
             font-family: "IBM Plex Sans", sans-serif; 
             font-size: 17px;
             -webkit-font-smoothing: antialiased;
           }
           
           @keyframes spin {
             from { transform: rotate(0deg); }
             to { transform: rotate(360deg); }
           }
           
           input[type=range] {
             -webkit-appearance: none;
             width: 100%;
             background: transparent;
             margin: 8px 0;
           }
           input[type=range]:focus {
             outline: none;
           }
           input[type=range]::-webkit-slider-runnable-track {
             width: 100%;
             height: 6px;
             cursor: pointer;
             background: #121212;
             border-radius: 4px;
             border: 1px solid #0d0d0d;
           }
           input[type=range]::-webkit-slider-thumb {
             height: 18px;
             width: 18px;
             border-radius: 50%;
             background: #38bdf8;
             cursor: pointer;
             -webkit-appearance: none;
             margin-top: -7px;
             box-shadow: none;
             transition: transform 0.15s ease, background 0.15s ease;
             border: 2px solid #050505;
           }
           input[type=range]::-webkit-slider-thumb:hover {
             transform: scale(1.25);
             background: #7dd3fc;
           }

           input[type=number] {
             background: #0b0b0b;
             border: 1px solid #1a1a1a;
             color: #f8fafc;
             border-radius: 8px;
             padding: 0.35rem 0.5rem;
             width: 88px;
             font-size: 0.9rem;
             font-family: "IBM Plex Mono", monospace;
           }
           input[type=number]:focus {
             outline: none;
             border-color: #38bdf8;
             box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.18);
           }
           
           ::-webkit-scrollbar { width: 8px; }
           ::-webkit-scrollbar-track { background: #0a0a0a; }
           ::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 4px; }
           ::-webkit-scrollbar-thumb:hover { background: #3f3f46; }

           .layout {
             display: grid;
             grid-template-columns: minmax(0, 1fr) 360px;
             gap: 2rem;
             align-items: start;
           }
           @media (max-width: 980px) {
             .layout {
               grid-template-columns: 1fr;
             }
           }
         `}</style>
      </Head>

      <div style={{ minHeight: '100vh', padding: '1.75rem 1.5rem', display: 'flex', justifyContent: 'center', position: 'relative' }}>
         {/* LOADING SCREEN */}
         {!modelReady && (
           <div style={{
             position: 'fixed',
             top: 0,
             left: 0,
             width: '100%',
             height: '100%',
             backgroundColor: 'rgba(0, 0, 0, 0.7)',
             backdropFilter: 'blur(8px)',
             display: 'flex',
             flexDirection: 'column',
             justifyContent: 'center',
             alignItems: 'center',
             zIndex: 9999,
           }}>
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '2rem',
              }}>
                {/* SPINNER */}
                <div style={{
                  width: '50px',
                  height: '50px',
                  border: '4px solid rgba(56, 189, 248, 0.2)',
                  borderTop: '4px solid #38bdf8',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite',
                }} />
                
                {/* LOADING TEXT */}
                <div style={{
                  textAlign: 'center',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '1rem',
                }}>
                  <h2 style={{
                    margin: 0,
                    fontSize: '1.5rem',
                    fontWeight: 600,
                    color: '#e5e7eb',
                  }}>
                    Loading Model
                  </h2>
                </div>
              </div>
           </div>
         )}

         <div style={{ width: '100%', maxWidth: '1100px' }}>
          
          {/* HEADER SECTION */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h1 style={{ margin: '0 0 0.3rem 0', fontSize: '1.9rem', fontWeight: 600, letterSpacing: '-0.015em' }}>
                Concrete Strength Estimator
              </h1>
            </div>
            <div style={{ fontSize: '0.8rem', color: '#6b7280', fontFamily: '"IBM Plex Mono", monospace' }}>
              v1.1
            </div>
          </div>

          <div className="layout">
            <div>
              {/* SECTION TABS */}
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                {[
                  { id: 'mix', label: 'Mix Design' },
                  { id: 'advanced', label: 'Advanced' },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveSection(tab.id)}
                    style={{
                      backgroundColor: activeSection === tab.id ? '#111827' : '#0a0a0a',
                      color: activeSection === tab.id ? '#e5e7eb' : '#9ca3af',
                      border: '1px solid #1a1a1a',
                      padding: '0.5rem 0.9rem',
                      borderRadius: '8px',
                      fontSize: '0.85rem',
                      cursor: 'pointer',
                    }}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {activeSection === 'mix' ? (
                <div style={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: '12px', padding: '1.25rem' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {MAIN_CONTROLS.map((input) => (
                      <div key={input.key} style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <label style={{ fontSize: '0.95rem', color: '#e5e7eb', fontWeight: 500 }}>{input.label}</label>
                          <input
                            type="number"
                            name={input.key}
                            min={input.min}
                            max={input.max}
                            step={input.key === 'water_TCM' ? 0.01 : input.step}
                            value={formData[input.key]}
                            onChange={handleChange}
                          />
                        </div>
                        <input
                          type="range"
                          name={input.key}
                          min={input.min}
                          max={input.max}
                          step={input.key === 'water_TCM' ? 0.01 : input.step}
                          value={formData[input.key]}
                          onChange={handleChange}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.1rem', fontSize: '0.8rem', color: '#6b7280', fontFamily: '"IBM Plex Mono", monospace' }}>
                          <span>{input.min}</span>
                          <span>{input.max}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div style={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: '12px', padding: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#a1a1aa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Advanced Parameters
                    </h3>
                    <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                      {categoryNames.map((name) => (
                        <button
                          key={name}
                          type="button"
                          onClick={() => setActiveCategory(name)}
                          style={{
                            backgroundColor: activeCategory === name ? '#111827' : '#0a0a0a',
                            color: activeCategory === name ? '#e5e7eb' : '#9ca3af',
                            border: '1px solid #1a1a1a',
                            padding: '0.35rem 0.7rem',
                            borderRadius: '8px',
                            fontSize: '0.8rem',
                            cursor: 'pointer',
                          }}
                        >
                          {name}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {ADVANCED_TABS[activeCategory].map((input) => (
                      <div key={input.key} style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <label style={{ fontSize: '0.95rem', color: '#e5e7eb', fontWeight: 500 }}>{input.label}</label>
                          <input
                            type="number"
                            name={input.key}
                            min={input.min}
                            max={input.max}
                            step={input.step}
                            value={formData[input.key]}
                            onChange={handleChange}
                          />
                        </div>
                        <input
                          type="range"
                          name={input.key}
                          min={input.min}
                          max={input.max}
                          step={input.step}
                          value={formData[input.key]}
                          onChange={handleChange}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.1rem', fontSize: '0.8rem', color: '#6b7280', fontFamily: '"IBM Plex Mono", monospace' }}>
                          <span>{input.min}</span>
                          <span>{input.max}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div>
              <div style={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: '14px', padding: '2rem' }}>
                <h2 style={{ margin: '0 0 1.25rem 0', fontSize: '1.05rem', fontWeight: 500, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                  Estimated Strength
                </h2>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', marginBottom: '1.25rem' }}>
                  <span style={{ fontSize: '3.4rem', fontWeight: 600, color: prediction ? '#e5e7eb' : '#52525b', lineHeight: 1 }}>
                    {prediction ? prediction.toFixed(2) : '--'}
                  </span>
                  <span style={{ fontSize: '1rem', color: '#6b7280', fontWeight: 500 }}>MPa</span>
                </div>
                <div style={{ height: '18px', backgroundColor: '#0b0b0b', borderRadius: '10px', border: '1px solid #1a1a1a', overflow: 'hidden', position: 'relative' }}>
                  <div style={{ height: '100%', width: `${strengthPercentage}%`, backgroundColor: '#38bdf8', transition: 'width 0.1s linear' }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.75rem', fontSize: '0.8rem', color: '#6b7280', fontWeight: 500, fontFamily: '"IBM Plex Mono", monospace' }}>
                  <span>0.00 MPa</span>
                  <span>120.00 MPa (Max)</span>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
