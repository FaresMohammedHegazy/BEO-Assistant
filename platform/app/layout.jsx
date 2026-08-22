import "./globals.css";

export const metadata = {
  title: "Aurelia Admin Platform",
  description: "Aurelia Hotels & Resorts — internal admin platform",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900">
        <nav className="border-b bg-white px-8 py-3 flex gap-6">
          <a href="/chat" className="font-medium hover:underline">Chat</a>
          <a href="/admin" className="font-medium hover:underline">Tools &amp; RAG</a>
          <a href="/admin/tickets" className="font-medium hover:underline">HITL &amp; Tickets</a>
        </nav>
        {children}
      </body>
    </html>
  );
}