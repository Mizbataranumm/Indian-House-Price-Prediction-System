import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./components/Dashboard";
import Predict from "./components/Predict";
import ModelLab from "./components/ModelLab";
import Analytics from "./components/Analytics";
import Compare from "./components/Compare";
import Calculators from "./components/Calculators";
import History from "./components/History";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="predict" element={<Predict />} />
          <Route path="models" element={<ModelLab />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="compare" element={<Compare />} />
          <Route path="calculators" element={<Calculators />} />
          <Route path="history" element={<History />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
