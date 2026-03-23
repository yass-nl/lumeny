// @ts-nocheck
import { Link, useLocation } from "react-router-dom";
import { useEffect } from "react";

const navStyle: React.CSSProperties = {
  position: "fixed",
  top: 0,
  left: 0,
  right: 0,
  zIndex: 100,
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "0 48px",
  height: 72,
  background: "rgba(6, 10, 16, 0.85)",
  backdropFilter: "blur(20px)",
  borderBottom: "1px solid rgba(79, 195, 247, 0.06)",
  transition: "all 0.3s ease",
};

const logoContainerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
  textDecoration: "none",
  color: "#e8eaf0",
};

const navLinksStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 36,
};

const navLinkStyle: React.CSSProperties = {
  color: "#a0a8b8",
  textDecoration: "none",
  fontSize: 11,
  fontFamily: "'DM Mono', monospace",
  letterSpacing: "0.08em",
  textTransform: "uppercase" as const,
  transition: "color 0.2s ease",
  cursor: "pointer",
};

const ctaButtonStyle: React.CSSProperties = {
  padding: "10px 24px",
  background: "transparent",
  border: "1px solid rgba(79, 195, 247, 0.3)",
  color: "#4fc3f7",
  fontSize: 11,
  fontFamily: "'DM Mono', monospace",
  letterSpacing: "0.08em",
  textTransform: "uppercase" as const,
  cursor: "pointer",
  borderRadius: 4,
  transition: "all 0.2s ease",
  textDecoration: "none",
};

const footerStyle: React.CSSProperties = {
  borderTop: "1px solid rgba(255,255,255,0.05)",
  padding: "48px 48px 32px",
  background: "#050810",
};

const footerGridStyle: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "2fr 1fr 1fr 1fr",
  gap: 48,
  maxWidth: 1200,
  margin: "0 auto",
  marginBottom: 48,
};

const footerColTitleStyle: React.CSSProperties = {
  fontFamily: "'Syne', sans-serif",
  fontSize: 11,
  fontWeight: 600,
  color: "#4fc3f7",
  letterSpacing: "0.12em",
  textTransform: "uppercase" as const,
  marginBottom: 20,
};

const footerLinkStyle: React.CSSProperties = {
  display: "block",
  color: "#6b7280",
  textDecoration: "none",
  fontSize: 12,
  fontFamily: "'DM Mono', monospace",
  marginBottom: 12,
  transition: "color 0.2s ease",
};

const footerBottomStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  paddingTop: 24,
  borderTop: "1px solid rgba(255,255,255,0.04)",
  maxWidth: 1200,
  margin: "0 auto",
};

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function Layout({ children }: { children: React.ReactNode }) {
  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div style={{ minHeight: "100vh", background: "#060a10" }}>
      <ScrollToTop />

      {/* Nav */}
      <nav style={navStyle}>
        <Link to="/" style={logoContainerStyle}>
          <img
            src="/logo-transparent.png"
            alt="LumenY"
            style={{ height: 30, opacity: 0.95 }}
          />
        </Link>

        <div style={navLinksStyle}>
          <a
            onClick={() => {
              if (window.location.pathname === "/") scrollTo("approach");
              else window.location.href = "/#approach";
            }}
            style={navLinkStyle}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#e8eaf0")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#a0a8b8")}
          >
            Approach
          </a>
          <a
            onClick={() => {
              if (window.location.pathname === "/") scrollTo("performance");
              else window.location.href = "/#performance";
            }}
            style={navLinkStyle}
            onMouseEnter={(e) => (e.currentTarget.style.color = "#e8eaf0")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "#a0a8b8")}
          >
            Performance
          </a>
          <a
            href="mailto:info@lumen-y.com"
            style={ctaButtonStyle}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(79, 195, 247, 0.08)";
              e.currentTarget.style.borderColor = "rgba(79, 195, 247, 0.5)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.borderColor = "rgba(79, 195, 247, 0.3)";
            }}
          >
            Get in Touch
          </a>
        </div>
      </nav>

      {/* Content */}
      <main>{children}</main>

      {/* Footer */}
      <footer style={footerStyle}>
        <div style={footerGridStyle}>
          <div>
            <img
              src="/logo-transparent.png"
              alt="LumenY"
              style={{ height: 26, opacity: 0.7, marginBottom: 16 }}
            />
            <p
              style={{
                color: "#4b5563",
                fontSize: 12,
                fontFamily: "'DM Mono', monospace",
                lineHeight: 1.7,
                maxWidth: 300,
              }}
            >
              Quantitative signals provider for institutional FX desks.
              Data-driven. Systematic. Independent.
            </p>
          </div>

          <div>
            <div style={footerColTitleStyle}>Company</div>
            <a
              onClick={() => {
                if (window.location.pathname === "/") scrollTo("approach");
                else window.location.href = "/#approach";
              }}
              style={footerLinkStyle}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#a0a8b8")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#6b7280")}
            >
              Approach
            </a>
            <a
              onClick={() => {
                if (window.location.pathname === "/") scrollTo("performance");
                else window.location.href = "/#performance";
              }}
              style={footerLinkStyle}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#a0a8b8")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#6b7280")}
            >
              Performance
            </a>
          </div>

          <div>
            <div style={footerColTitleStyle}>Legal</div>
            <Link
              to="/privacy"
              style={footerLinkStyle}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#a0a8b8")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#6b7280")}
            >
              Privacy Policy
            </Link>
            <Link
              to="/terms"
              style={footerLinkStyle}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#a0a8b8")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#6b7280")}
            >
              Terms of Service
            </Link>
            <Link
              to="/risk-disclosure"
              style={footerLinkStyle}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#a0a8b8")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#6b7280")}
            >
              Risk Disclosure
            </Link>
          </div>

          <div>
            <div style={footerColTitleStyle}>Contact</div>
            <a
              href="mailto:info@lumen-y.com"
              style={footerLinkStyle}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#a0a8b8")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#6b7280")}
            >
              info@lumen-y.com
            </a>
          </div>
        </div>

        <div style={footerBottomStyle}>
          <span
            style={{
              color: "#374151",
              fontSize: 11,
              fontFamily: "'DM Mono', monospace",
            }}
          >
            &copy; {new Date().getFullYear()} LumenY. All rights reserved.
          </span>
          <span
            style={{
              color: "#374151",
              fontSize: 10,
              fontFamily: "'DM Mono', monospace",
              maxWidth: 500,
              textAlign: "right",
              lineHeight: 1.5,
            }}
          >
            LumenY provides informational signals only. Not financial advice.
            Past performance does not guarantee future results.
          </span>
        </div>
      </footer>
    </div>
  );
}
