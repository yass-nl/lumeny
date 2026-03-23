// @ts-nocheck
import Layout from "../components/Layout";

const sectionStyle: React.CSSProperties = {
  marginBottom: 40,
};

const headingStyle: React.CSSProperties = {
  fontFamily: "'Syne', sans-serif",
  fontSize: 16,
  fontWeight: 600,
  color: "#e8eaf0",
  marginBottom: 16,
};

const textStyle: React.CSSProperties = {
  fontFamily: "'DM Mono', monospace",
  fontSize: 12,
  color: "#6b7280",
  lineHeight: 1.8,
  marginBottom: 12,
};

export default function PrivacyPolicy() {
  return (
    <Layout>
      <div
        style={{
          maxWidth: 720,
          margin: "0 auto",
          padding: "140px 48px 120px",
        }}
      >
        <div
          style={{
            fontSize: 10,
            fontFamily: "'DM Mono', monospace",
            color: "#4fc3f7",
            letterSpacing: "0.15em",
            textTransform: "uppercase",
            marginBottom: 16,
          }}
        >
          Legal
        </div>
        <h1
          style={{
            fontFamily: "'Syne', sans-serif",
            fontSize: 36,
            fontWeight: 700,
            color: "#e8eaf0",
            marginBottom: 12,
          }}
        >
          Privacy Policy
        </h1>
        <p
          style={{
            fontFamily: "'DM Mono', monospace",
            fontSize: 11,
            color: "#374151",
            marginBottom: 64,
          }}
        >
          Last updated: March 2026
        </p>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>1. Introduction</h2>
          <p style={textStyle}>
            LumenY ("we", "our", "us") operates the website lumen-y.com. This
            Privacy Policy explains how we collect, use, and protect information
            when you interact with our website and services. We are committed to
            safeguarding the privacy of our visitors and clients.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>2. Information We Collect</h2>
          <p style={textStyle}>
            <strong style={{ color: "#a0a8b8" }}>
              Information you provide directly:
            </strong>{" "}
            When you contact us via email or request information about our
            services, we may collect your name, email address, company name,
            and any other information you choose to share.
          </p>
          <p style={textStyle}>
            <strong style={{ color: "#a0a8b8" }}>
              Automatically collected information:
            </strong>{" "}
            We may collect standard web analytics data such as IP address,
            browser type, pages visited, and time spent on the site. This data
            is used solely for improving the user experience and understanding
            site traffic.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>3. How We Use Your Information</h2>
          <p style={textStyle}>
            We use the information we collect to: respond to inquiries and
            provide our services; communicate with you regarding our signal
            products; improve our website and services; comply with legal
            obligations. We do not sell, rent, or trade your personal
            information to third parties.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>4. Data Security</h2>
          <p style={textStyle}>
            We implement appropriate technical and organizational measures to
            protect your personal information against unauthorized access,
            alteration, disclosure, or destruction. However, no method of
            transmission over the internet is 100% secure, and we cannot
            guarantee absolute security.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>5. Third-Party Services</h2>
          <p style={textStyle}>
            Our website may use third-party analytics services to help us
            understand usage patterns. These services may collect information
            sent by your browser as part of a web page request. Their use of
            this information is governed by their own privacy policies.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>6. Data Retention</h2>
          <p style={textStyle}>
            We retain personal information only for as long as necessary to
            fulfill the purposes for which it was collected, or as required by
            applicable law. When information is no longer needed, it is securely
            deleted or anonymized.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>7. Your Rights</h2>
          <p style={textStyle}>
            You have the right to access, correct, or delete your personal
            information. You may also object to or restrict certain processing
            activities. To exercise any of these rights, please contact us at
            info@lumen-y.com.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>8. Changes to This Policy</h2>
          <p style={textStyle}>
            We may update this Privacy Policy from time to time. Any changes
            will be posted on this page with an updated revision date. We
            encourage you to review this policy periodically.
          </p>
        </div>

        <div style={sectionStyle}>
          <h2 style={headingStyle}>9. Contact</h2>
          <p style={textStyle}>
            For questions about this Privacy Policy, contact us at:{" "}
            <a
              href="mailto:info@lumen-y.com"
              style={{ color: "#4fc3f7", textDecoration: "none" }}
            >
              info@lumen-y.com
            </a>
          </p>
        </div>
      </div>
    </Layout>
  );
}
