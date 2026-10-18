import { Suspense } from "react";

import ReviewWorkspace from "@/components/ReviewWorkspace";
import { Empty } from "@/components/ui";

async function Loader({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ReviewWorkspace id={id} />;
}

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<Empty>Loading review...</Empty>}>
      <Loader params={params} />
    </Suspense>
  );
}
