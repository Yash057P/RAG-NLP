import React from "react";
import ReactDOM from "react-dom/client";
import { Toaster } from "sonner";

import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
    <Toaster
      position="bottom-right"
      closeButton
      richColors
      toastOptions={{
        classNames: {
          toast: "!bg-popover/95 !text-popover-foreground !border-border !backdrop-blur-xl",
          description: "!text-muted-foreground",
        },
      }}
    />
  </React.StrictMode>,
);
