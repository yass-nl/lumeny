// @ts-nocheck
import { useNavigate } from "react-router-dom";

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div style={{
      minHeight: "100vh",
      background: "#060a10",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'Syne', sans-serif",
      color: "#e8eaf0"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');
      `}</style>
      {/* Placeholder — full landing page coming soon */}
      <div style={{ textAlign: "center" }}>
        <img src="/logo-transparent.png" style={{ height: "40px", marginBottom: "32px" }} alt="LumenY" />
        <p style={{
          fontSize: "13px",
          fontFamily: "'DM Mono', monospace",
          color: "rgba(255,255,255,0.25)",
          marginBottom: "32px"
        }}>Landing page coming soon</p>
        <button
          onClick={() => navigate("/login")}
          style={{
            padding: "10px 24px",
            background: "rgba(79,195,247,0.1)",
            border: "1px solid rgba(79,195,247,0.25)",
            borderRadius: "8px",
            color: "#4fc3f7",
            fontSize: "13px",
            fontFamily: "'Syne', sans-serif",
            cursor: "pointer",
            fontWeight: "600"
          }}
        >Go to Dashboard →</button>
      </div>
    </div>
  );
}