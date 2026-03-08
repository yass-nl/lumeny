// @ts-nocheck
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    // Temporary: just set a flag and redirect to dashboard
    // Replace with real Supabase auth later
    localStorage.setItem("lumeny_auth", "true");
    navigate("/dashboard");
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#060a10",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'Syne', sans-serif"
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700;800&display=swap');
        .auth-input:focus { outline: none; border-color: rgba(79,195,247,0.4) !important; }
        .auth-btn:hover { background: rgba(79,195,247,0.15) !important; }
      `}</style>

      <div style={{
        width: "100%",
        maxWidth: "380px",
        padding: "0 24px"
      }}>
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: "40px" }}>
          <img src="/logo-transparent.png" style={{ height: "32px", width: "auto" }} alt="LumenY" />
        </div>

        {/* Card */}
        <div style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: "16px",
          padding: "32px"
        }}>
          <h2 style={{
            fontSize: "18px",
            fontWeight: "700",
            color: "#e8eaf0",
            marginBottom: "6px"
          }}>Welcome back</h2>
          <p style={{
            fontSize: "13px",
            color: "rgba(255,255,255,0.35)",
            marginBottom: "28px",
            fontFamily: "'DM Mono', monospace"
          }}>Sign in to your account</p>

          <form onSubmit={handleLogin}>
            <div style={{ marginBottom: "16px" }}>
              <label style={{
                fontSize: "11px",
                fontFamily: "'DM Mono', monospace",
                color: "rgba(255,255,255,0.3)",
                letterSpacing: "0.08em",
                display: "block",
                marginBottom: "6px"
              }}>EMAIL</label>
              <input
                className="auth-input"
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@example.com"
                style={{
                  width: "100%",
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "8px",
                  padding: "10px 14px",
                  color: "#e8eaf0",
                  fontSize: "13px",
                  fontFamily: "'Syne', sans-serif",
                  transition: "border-color 0.2s",
                  boxSizing: "border-box"
                }}
              />
            </div>

            <div style={{ marginBottom: "24px" }}>
              <label style={{
                fontSize: "11px",
                fontFamily: "'DM Mono', monospace",
                color: "rgba(255,255,255,0.3)",
                letterSpacing: "0.08em",
                display: "block",
                marginBottom: "6px"
              }}>PASSWORD</label>
              <input
                className="auth-input"
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                style={{
                  width: "100%",
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  borderRadius: "8px",
                  padding: "10px 14px",
                  color: "#e8eaf0",
                  fontSize: "13px",
                  fontFamily: "'Syne', sans-serif",
                  transition: "border-color 0.2s",
                  boxSizing: "border-box"
                }}
              />
            </div>

            <button
              type="submit"
              className="auth-btn"
              style={{
                width: "100%",
                padding: "12px",
                background: "rgba(79,195,247,0.1)",
                border: "1px solid rgba(79,195,247,0.25)",
                borderRadius: "8px",
                color: "#4fc3f7",
                fontSize: "13px",
                fontWeight: "600",
                fontFamily: "'Syne', sans-serif",
                cursor: "pointer",
                letterSpacing: "0.04em",
                transition: "background 0.2s"
              }}
            >Sign in</button>
          </form>
        </div>

        <p style={{
          textAlign: "center",
          marginTop: "20px",
          fontSize: "12px",
          fontFamily: "'DM Mono', monospace",
          color: "rgba(255,255,255,0.2)"
        }}>
          Don't have an account?{" "}
          <span style={{ color: "#4fc3f7", cursor: "pointer" }}>
            Join the waitlist
          </span>
        </p>
      </div>
    </div>
  );
}