import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surface hierarchy (dark navy system)
        surface: {
          base: "#0e1322",
          dim: "#0e1322",
          low: "#161b2b",
          DEFAULT: "#1a1f2f",
          high: "#25293a",
          highest: "#2f3445",
          bright: "#343949",
          lowest: "#090e1c",
        },
        // Primary indigo
        primary: {
          DEFAULT: "#c0c1ff",
          container: "#8083ff",
          fixed: "#e1e0ff",
          "fixed-dim": "#c0c1ff",
          on: "#1000a9",
          "on-container": "#0d0096",
        },
        // Secondary violet
        secondary: {
          DEFAULT: "#d0bcff",
          container: "#571bc1",
          fixed: "#e9ddff",
          "fixed-dim": "#d0bcff",
          on: "#3c0091",
          "on-container": "#c4abff",
        },
        // Tertiary cyan
        tertiary: {
          DEFAULT: "#2fd9f4",
          container: "#008395",
          fixed: "#a2eeff",
          "fixed-dim": "#2fd9f4",
          on: "#00363e",
          "on-container": "#000608",
        },
        // Text
        "on-surface": "#dee1f7",
        "on-surface-variant": "#c7c4d7",
        outline: "#908fa0",
        "outline-variant": "#464554",
        // Semantic
        error: "#ffb4ab",
        "error-container": "#93000a",
        "on-error": "#690005",
        success: "#4ade80",
        warning: "#fbbf24",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Consolas", "monospace"],
      },
      boxShadow: {
        glow: "0 0 32px 4px rgba(192, 193, 255, 0.08)",
        "glow-primary": "0 0 20px 2px rgba(99, 102, 241, 0.3)",
        "glow-tertiary": "0 0 20px 2px rgba(47, 217, 244, 0.25)",
        panel: "0 8px 32px rgba(9, 14, 28, 0.6)",
        card: "0 2px 16px rgba(9, 14, 28, 0.4)",
      },
      backgroundImage: {
        "gradient-primary": "linear-gradient(135deg, #c0c1ff 0%, #8083ff 100%)",
        "gradient-secondary": "linear-gradient(135deg, #d0bcff 0%, #571bc1 100%)",
        "gradient-tertiary": "linear-gradient(135deg, #2fd9f4 0%, #008395 100%)",
        "surface-noise":
          "radial-gradient(circle at 20% 20%, rgba(192,193,255,0.04) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(47,217,244,0.03) 0%, transparent 50%)",
      },
      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.5rem",
        "4xl": "2rem",
      },
      animation: {
        "pulse-slow": "pulse 3s ease-in-out infinite",
        float: "float 6s ease-in-out infinite",
        shimmer: "shimmer 1.5s linear infinite",
        "ring-fill": "ring-fill 0.8s ease-out forwards",
        "fade-up": "fade-up 0.3s ease-out forwards",
        "slide-in": "slide-in 0.25s ease-out forwards",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-6px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "ring-fill": {
          "0%": { strokeDashoffset: "283" },
          "100%": { strokeDashoffset: "var(--target-dashoffset)" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in": {
          "0%": { opacity: "0", transform: "translateX(-8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
