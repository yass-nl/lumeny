// @ts-nocheck
import { useState, useEffect, useRef } from "react";

const TIMEFRAMES = ["5m", "15m", "1H", "4H", "1D"];
const FEATURES = ["Regime", "Momentum", "Confluence", "Volatility", "Structure"];

function WaveformCanvas() {
  const canvasRef = useRef(null);
  const frameRef = useRef(0);
  const pointsRef = useRef(Array(80).fill(0));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let animId;
    let t = 0;

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      // Shift points left
      pointsRef.current.shift();
      const newVal = 0.5 + 0.3 * Math.sin(t * 0.08) + 0.15 * Math.sin(t * 0.21) + 0.05 * (Math.random() - 0.5);
      pointsRef.current.push(Math.max(0.05, Math.min(0.95, newVal)));
      t++;

      const pts = pointsRef.current;
      const step = W / (pts.length - 1);

      // Gradient fill under line
      const grad = ctx.createLinearGradient(0, 0, 0, H);
      grad.addColorStop(0, "rgba(79,195,247,0.15)");
      grad.addColorStop(1, "rgba(79,195,247,0)");

      ctx.beginPath();
      ctx.moveTo(0, H);
      pts.forEach((p, i) => {
        ctx.lineTo(i * step, H - p * H);
      });
      ctx.lineTo(W, H);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // Line
      ctx.beginPath();
      pts.forEach((p, i) => {
        if (i === 0) ctx.moveTo(0, H - p * H);
        else ctx.lineTo(i * step, H - p * H);
      });
      ctx.strokeStyle = "rgba(79,195,247,0.6)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Glowing head dot
      const lastX = (pts.length - 1) * step;
      const lastY = H - pts[pts.length - 1] * H;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#4fc3f7";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(lastX, lastY, 6, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(79,195,247,0.2)";
      ctx.fill();

      animId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animId);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={200}
      height={40}
      style={{ width: "200px", height: "40px", display: "block" }}
    />
  );
}

export default function ModelActivity({ pair }: { pair: string }) {
  const [activeTf, setActiveTf] = useState(0);
  const [activeFeature, setActiveFeature] = useState(0);
  const [featureValues, setFeatureValues] = useState(
    FEATURES.map(() => Math.floor(Math.random() * 40 + 55))
  );
  const [computedAgo, setComputedAgo] = useState(0);
  const [scanPulse, setScanPulse] = useState(false);

  // Cycle timeframes
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveTf(tf => {
        const next = (tf + 1) % TIMEFRAMES.length;
        if (next === 0) setScanPulse(p => !p);
        return next;
      });
    }, 800);
    return () => clearInterval(interval);
  }, []);

  // Cycle features
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature(f => (f + 1) % FEATURES.length);
      setFeatureValues(vals =>
        vals.map((v, i) =>
          i === activeFeature
            ? Math.floor(Math.random() * 40 + 55)
            : v
        )
      );
    }, 1200);
    return () => clearInterval(interval);
  }, [activeFeature]);

  // Tick computed ago
  useEffect(() => {
    setComputedAgo(0);
    const interval = setInterval(() => {
      setComputedAgo(s => s + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [pair]);

  return (
    <div style={{
      background: "rgba(255,255,255,0.02)",
      border: "1px solid rgba(255,255,255,0.05)",
      borderRadius: "10px",
      padding: "10px 16px",
      display: "flex",
      alignItems: "center",
      gap: "20px",
      marginTop: "10px",
      flexShrink: 0,
      overflow: "hidden"
    }}>

      {/* Status indicator */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
        <div style={{ position: "relative", width: "8px", height: "8px" }}>
          <div style={{
            width: "8px", height: "8px", borderRadius: "50%",
            background: "#4fc3f7",
            boxShadow: "0 0 6px #4fc3f7",
            animation: "modelPulse 1.5s ease-in-out infinite"
          }} />
        </div>
        <span style={{
          fontSize: "9px", fontFamily: "'DM Mono', monospace",
          color: "rgba(255,255,255,0.3)", letterSpacing: "0.1em",
          textTransform: "uppercase", whiteSpace: "nowrap"
        }}>Model Active</span>
      </div>

      {/* Divider */}
      <div style={{ width: "1px", height: "28px", background: "rgba(255,255,255,0.06)", flexShrink: 0 }} />

      {/* Timeframe scanner */}
      <div style={{ display: "flex", alignItems: "center", gap: "6px", flexShrink: 0 }}>
        <span style={{
          fontSize: "9px", fontFamily: "'DM Mono', monospace",
          color: "rgba(255,255,255,0.2)", letterSpacing: "0.08em",
          marginRight: "4px", whiteSpace: "nowrap"
        }}>SCANNING</span>
        {TIMEFRAMES.map((tf, i) => (
          <div key={tf} style={{
            padding: "3px 7px",
            borderRadius: "4px",
            fontSize: "10px",
            fontFamily: "'DM Mono', monospace",
            fontWeight: "500",
            transition: "all 0.3s ease",
            background: i === activeTf ? "rgba(79,195,247,0.15)" : "transparent",
            color: i === activeTf ? "#4fc3f7" : "rgba(255,255,255,0.15)",
            border: `1px solid ${i === activeTf ? "rgba(79,195,247,0.3)" : "transparent"}`,
            boxShadow: i === activeTf ? "0 0 8px rgba(79,195,247,0.2)" : "none",
            transform: i === activeTf ? "scale(1.05)" : "scale(1)"
          }}>{tf}</div>
        ))}
      </div>

      {/* Divider */}
      <div style={{ width: "1px", height: "28px", background: "rgba(255,255,255,0.06)", flexShrink: 0 }} />

      {/* Feature evaluator */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px", flexShrink: 0 }}>
        <span style={{
          fontSize: "9px", fontFamily: "'DM Mono', monospace",
          color: "rgba(255,255,255,0.2)", letterSpacing: "0.08em",
          whiteSpace: "nowrap"
        }}>EVALUATING</span>
        {FEATURES.map((f, i) => (
          <div key={f} style={{
            display: "flex", alignItems: "center", gap: "4px",
            transition: "opacity 0.3s",
            opacity: i === activeFeature ? 1 : 0.25
          }}>
            <div style={{
              width: "5px", height: "5px", borderRadius: "50%",
              background: i === activeFeature ? "#4fc3f7" : "rgba(255,255,255,0.2)",
              boxShadow: i === activeFeature ? "0 0 6px #4fc3f7" : "none",
              transition: "all 0.3s",
              flexShrink: 0
            }} />
            <span style={{
              fontSize: "9px", fontFamily: "'DM Mono', monospace",
              color: i === activeFeature ? "rgba(255,255,255,0.7)" : "rgba(255,255,255,0.2)",
              whiteSpace: "nowrap", transition: "color 0.3s"
            }}>{f}</span>
          </div>
        ))}
      </div>

      {/* Divider */}
      <div style={{ width: "1px", height: "28px", background: "rgba(255,255,255,0.06)", flexShrink: 0 }} />

      {/* Waveform */}
      <div style={{ flexShrink: 0 }}>
        <WaveformCanvas />
      </div>

      {/* Divider */}
      <div style={{ width: "1px", height: "28px", background: "rgba(255,255,255,0.06)", flexShrink: 0 }} />

      {/* Last computed */}
      <div style={{ flexShrink: 0, marginLeft: "auto" }}>
        <div style={{
          fontSize: "9px", fontFamily: "'DM Mono', monospace",
          color: "rgba(255,255,255,0.2)", letterSpacing: "0.08em",
          marginBottom: "2px"
        }}>LAST COMPUTED</div>
        <div style={{
          fontSize: "11px", fontFamily: "'DM Mono', monospace",
          color: "rgba(255,255,255,0.45)"
        }}>
          {computedAgo === 0 ? "just now" : `${computedAgo}s ago`}
        </div>
      </div>

      <style>{`
        @keyframes modelPulse {
          0%, 100% { opacity: 0.6; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>
    </div>
  );
}