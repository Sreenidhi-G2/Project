import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          980: "#080b0f",
          950: "#0d1117",
          925: "#111821",
          900: "#151c25",
          850: "#1a2430",
          800: "#212c39",
          700: "#334152"
        },
        signal: {
          real: "#33d69f",
          review: "#f2b84b",
          fake: "#f05f5f",
          info: "#56b7ff",
          cyan: "#32d3ee"
        }
      },
      boxShadow: {
        panel: "0 18px 42px rgba(0, 0, 0, 0.28)",
        insetline: "inset 0 1px 0 rgba(255, 255, 255, 0.04)"
      }
    },
  },
  plugins: [],
} satisfies Config;
