// @ts-nocheck
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const WEB3FORMS_KEY = "381a58be-08d5-4c1d-acdb-ee6b2aa0383f";

const overlayStyle: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  zIndex: 200,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "rgba(0, 0, 0, 0.7)",
  backdropFilter: "blur(8px)",
};

const modalStyle: React.CSSProperties = {
  background: "#0a0f18",
  border: "1px solid rgba(79, 195, 247, 0.1)",
  borderRadius: 10,
  padding: "40px 36px",
  width: "100%",
  maxWidth: 480,
  position: "relative",
};

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: 10,
  fontFamily: "'DM Mono', monospace",
  color: "#4fc3f7",
  letterSpacing: "0.12em",
  textTransform: "uppercase" as const,
  marginBottom: 8,
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "12px 14px",
  background: "rgba(255,255,255,0.03)",
  border: "1px solid rgba(255,255,255,0.07)",
  borderRadius: 4,
  color: "#e8eaf0",
  fontSize: 13,
  fontFamily: "'DM Mono', monospace",
  outline: "none",
  transition: "border-color 0.2s ease",
};

const inputFocusColor = "rgba(79, 195, 247, 0.25)";

function Input({
  label,
  ...props
}: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div style={{ marginBottom: 20 }}>
      <label style={labelStyle}>{label}</label>
      <input
        style={inputStyle}
        onFocus={(e) => (e.currentTarget.style.borderColor = inputFocusColor)}
        onBlur={(e) =>
          (e.currentTarget.style.borderColor = "rgba(255,255,255,0.07)")
        }
        {...props}
      />
    </div>
  );
}

export default function ContactModal({
  isOpen,
  onClose,
  subject,
}: {
  isOpen: boolean;
  onClose: () => void;
  subject: string;
}) {
  const [form, setForm] = useState({
    name: "",
    company: "",
    email: "",
    message: "",
  });
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">(
    "idle"
  );

  const update = (field: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("sending");

    try {
      const res = await fetch("https://api.web3forms.com/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          access_key: WEB3FORMS_KEY,
          subject: `LumenY — ${subject}`,
          from_name: `${form.name} (${form.company})`,
          name: form.name,
          company: form.company,
          email: form.email,
          message: form.message,
        }),
      });

      if (res.ok) {
        setStatus("sent");
        setTimeout(() => {
          onClose();
          setStatus("idle");
          setForm({ name: "", company: "", email: "", message: "" });
        }, 2000);
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          style={overlayStyle}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={(e) => {
            if (e.target === e.currentTarget) onClose();
          }}
        >
          <motion.div
            className="modal-container"
            style={modalStyle}
            initial={{ opacity: 0, y: 20, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.97 }}
            transition={{ duration: 0.25, ease: [0.25, 0.46, 0.45, 0.94] }}
          >
            {/* Close button */}
            <button
              onClick={onClose}
              style={{
                position: "absolute",
                top: 16,
                right: 16,
                background: "none",
                border: "none",
                color: "#374151",
                fontSize: 18,
                cursor: "pointer",
                fontFamily: "'DM Mono', monospace",
                transition: "color 0.2s ease",
                lineHeight: 1,
                padding: 4,
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#a0a8b8")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#374151")}
            >
              ✕
            </button>

            <div
              style={{
                fontSize: 10,
                fontFamily: "'DM Mono', monospace",
                color: "#4fc3f7",
                letterSpacing: "0.15em",
                textTransform: "uppercase",
                marginBottom: 8,
              }}
            >
              {subject}
            </div>
            <h2
              style={{
                fontFamily: "'Syne', sans-serif",
                fontSize: 24,
                fontWeight: 700,
                color: "#e8eaf0",
                marginBottom: 8,
              }}
            >
              {subject === "Schedule a Call"
                ? "Let's talk"
                : "Get in touch"}
            </h2>
            <p
              style={{
                fontFamily: "'DM Mono', monospace",
                fontSize: 12,
                color: "#4b5563",
                lineHeight: 1.6,
                marginBottom: 32,
              }}
            >
              {subject === "Schedule a Call"
                ? "Leave your details and we'll arrange a call at your convenience."
                : "Send us your question and we'll get back to you shortly."}
            </p>

            {status === "sent" ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                style={{
                  textAlign: "center",
                  padding: "40px 0",
                }}
              >
                <div
                  style={{
                    fontSize: 32,
                    marginBottom: 16,
                    color: "#4fc3f7",
                  }}
                >
                  ✓
                </div>
                <div
                  style={{
                    fontFamily: "'Syne', sans-serif",
                    fontSize: 18,
                    fontWeight: 600,
                    color: "#e8eaf0",
                    marginBottom: 8,
                  }}
                >
                  Message sent
                </div>
                <div
                  style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: 12,
                    color: "#6b7280",
                  }}
                >
                  We'll be in touch.
                </div>
              </motion.div>
            ) : (
              <form onSubmit={handleSubmit}>
                <div
                  className="modal-form-row"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 16,
                  }}
                >
                  <Input
                    label="Name"
                    placeholder="Your name"
                    value={form.name}
                    onChange={update("name")}
                    required
                  />
                  <Input
                    label="Company"
                    placeholder="Your firm"
                    value={form.company}
                    onChange={update("company")}
                    required
                  />
                </div>
                <Input
                  label="Email"
                  type="email"
                  placeholder="you@company.com"
                  value={form.email}
                  onChange={update("email")}
                  required
                />
                <div style={{ marginBottom: 24 }}>
                  <label style={labelStyle}>Message</label>
                  <textarea
                    style={{
                      ...inputStyle,
                      resize: "vertical",
                      minHeight: 100,
                    }}
                    placeholder={
                      subject === "Schedule a Call"
                        ? "Preferred time, timezone, or anything you'd like to discuss..."
                        : "Your question or inquiry..."
                    }
                    value={form.message}
                    onChange={update("message")}
                    onFocus={(e) =>
                      (e.currentTarget.style.borderColor = inputFocusColor)
                    }
                    onBlur={(e) =>
                      (e.currentTarget.style.borderColor =
                        "rgba(255,255,255,0.07)")
                    }
                    required
                  />
                </div>

                {status === "error" && (
                  <div
                    style={{
                      fontFamily: "'DM Mono', monospace",
                      fontSize: 11,
                      color: "#ff4757",
                      marginBottom: 16,
                    }}
                  >
                    Something went wrong. Please try again or email us
                    directly at info@lumen-y.com.
                  </div>
                )}

                <button
                  type="submit"
                  disabled={status === "sending"}
                  style={{
                    width: "100%",
                    padding: "14px",
                    background:
                      status === "sending"
                        ? "rgba(79, 195, 247, 0.05)"
                        : "rgba(79, 195, 247, 0.1)",
                    border: "1px solid rgba(79, 195, 247, 0.3)",
                    color: "#4fc3f7",
                    fontSize: 12,
                    fontFamily: "'DM Mono', monospace",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase" as const,
                    cursor: status === "sending" ? "default" : "pointer",
                    borderRadius: 4,
                    transition: "all 0.25s ease",
                  }}
                  onMouseEnter={(e) => {
                    if (status !== "sending") {
                      e.currentTarget.style.background =
                        "rgba(79, 195, 247, 0.18)";
                      e.currentTarget.style.borderColor =
                        "rgba(79, 195, 247, 0.5)";
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background =
                      "rgba(79, 195, 247, 0.1)";
                    e.currentTarget.style.borderColor =
                      "rgba(79, 195, 247, 0.3)";
                  }}
                >
                  {status === "sending" ? "Sending..." : "Send"}
                </button>
              </form>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
