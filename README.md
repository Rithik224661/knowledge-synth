# Knowledge Synthesis Platform

A modern web application for research, knowledge management, and synthesis. This platform helps users conduct research queries, visualize data, and synthesize information from multiple sources.

## Features

- **Research Query Interface**: Submit research questions and get comprehensive answers
- **Data Visualization**: View research trends, tables, and visual representations of data
- **Source Management**: Track and cite sources used in research
- **Evaluation Metrics**: Assess the quality and relevance of research results
- **Recommendations**: Get suggestions for further research based on your queries

## Project Structure

The project consists of two main components:

### Frontend (React + TypeScript)
- Modern UI built with React, TypeScript, and Tailwind CSS
- Visualization components for displaying research data
- Context-based state management

### Backend (Python + FastAPI)
- Research processing engine
- API endpoints for research queries and status tracking
- Data processing and synthesis capabilities

## Setup Instructions

### Frontend Setup

1. Clone the repository
2. Install dependencies: `npm install`
3. Start the development server: `npm run dev`
4. Open [http://localhost:5173](http://localhost:5173) to view it in your browser

### Backend Setup

1. Navigate to the backend directory: `cd backend`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Start the backend server: `uvicorn app.main:app --reload --port 8000`

## Available Scripts

### Frontend
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Backend
- `uvicorn app.main:app --reload --port 8000` - Start the backend server with hot reloading

## Deployment

### Frontend
Build the project with `npm run build` and deploy the contents of the `dist` directory to your preferred hosting provider.

### Backend
Deploy the FastAPI application using a WSGI server like Gunicorn or using cloud services that support Python applications.
